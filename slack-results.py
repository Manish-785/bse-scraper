import os
import time
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf
import requests
import fitz
import openpyxl
import re
import logging
import traceback
from logging.handlers import RotatingFileHandler
import signal
import sys
from pathlib import Path
import threading
from functools import wraps

# --- Enhanced Logging Setup ---
def setup_logging():
    """Setup comprehensive logging with rotation and multiple handlers"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler for all logs (rotating)
    file_handler = RotatingFileHandler(
        'logs/bse_scraper.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # File handler for errors only
    error_handler = RotatingFileHandler(
        'logs/bse_scraper_errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# --- Health Check and Monitoring ---
class HealthMonitor:
    def __init__(self):
        self.last_successful_run = datetime.now()
        self.consecutive_failures = 0
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = datetime.now()
        
    def record_success(self):
        self.last_successful_run = datetime.now()
        self.consecutive_failures = 0
        self.total_processed += 1
        
    def record_failure(self):
        self.consecutive_failures += 1
        self.total_errors += 1
        
    def get_health_status(self):
        uptime = datetime.now() - self.start_time
        time_since_success = datetime.now() - self.last_successful_run
        
        return {
            'uptime': str(uptime),
            'last_successful_run': self.last_successful_run.strftime('%Y-%m-%d %H:%M:%S'),
            'time_since_success': str(time_since_success),
            'consecutive_failures': self.consecutive_failures,
            'total_processed': self.total_processed,
            'total_errors': self.total_errors,
            'error_rate': f"{(self.total_errors / max(self.total_processed, 1)) * 100:.2f}%"
        }

health_monitor = HealthMonitor()

# --- Decorator for retry logic ---
def retry_on_failure(max_retries=3, delay=5, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"{func.__name__} failed on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
                        health_monitor.record_failure()
            raise last_exception
        return wrapper
    return decorator

# --- Graceful Shutdown Handler ---
class GracefulShutdown:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown = True
        
    def should_shutdown(self):
        return self.shutdown

shutdown_handler = GracefulShutdown()

# --- Enhanced Summarization with Better Error Handling ---
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

PROMPT_TEMPLATE = (
    "You are a financial analyst. Read the following BSE company update and write a one-line summary focusing only on the most important detail. "
    "Do not include any introductory or closing phrases, and do not write anything except the summary itself.\n\n{text}\n"
)

@retry_on_failure(max_retries=3, delay=3)
def summarise_with_groq_model(text, model, stream=False):
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found")
        return None
        
    logger.debug(f"Attempting summarization with Groq model: {model}")
    client = Groq(api_key=GROQ_API_KEY)
    prompt = PROMPT_TEMPLATE.format(text=text)
    
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        stream=stream
    )
    
    if stream:
        result = ""
        for chunk in completion:
            part = chunk.choices[0].delta.content or ""
            result += part
        return result.strip()
    else:
        return completion.choices[0].message.content.strip()

@retry_on_failure(max_retries=3, delay=5)
def summarise_with_gemini(text):
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found")
        return None
        
    logger.debug("Attempting summarization with Gemini")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    prompt = PROMPT_TEMPLATE.format(text=text)
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    
    if response.status_code == 429:
        logger.warning("Gemini rate limit hit")
        time.sleep(10)
        raise Exception("Rate limit exceeded")
        
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

@retry_on_failure(max_retries=5, delay=10)
def summarise_with_openrouter(text, api_key=OPENROUTER_API_KEY):
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found")
        return None
        
    logger.debug("Attempting summarization with OpenRouter")
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    prompt = PROMPT_TEMPLATE.format(text=text)
    
    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528:free",
        messages=[{"role": "user", "content": prompt}],
        timeout=30
    )
    return completion.choices[0].message.content

def summarise_bse_text(text, stream=False):
    """Enhanced summarization with comprehensive error handling"""
    logger.info("Starting text summarization")
    
    groq_models = [
        "llama3-70b-8192",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant",
    ]
    
    # Try Groq models
    for model in groq_models:
        try:
            logger.debug(f"Trying Groq model: {model}")
            summary = summarise_with_groq_model(text, model, stream=stream)
            if summary:
                logger.info(f"✔ Successfully summarized with {model}")
                return summary
        except Exception as e:
            logger.warning(f"✘ Failed with {model}: {e}")
            continue

    # Fallback to Gemini
    try:
        logger.debug("Falling back to Gemini")
        summary = summarise_with_gemini(text)
        if summary:
            logger.info("✔ Successfully summarized with Gemini")
            return summary
    except Exception as e:
        logger.warning(f"Gemini failed: {e}")

    # Fallback to OpenRouter
    try:
        logger.debug("Falling back to OpenRouter")
        summary = summarise_with_openrouter(text)
        if summary:
            logger.info("✔ Successfully summarized with OpenRouter")
            return summary
    except Exception as e:
        logger.warning(f"OpenRouter failed: {e}")

    logger.error("All summarization attempts failed")
    return "Summary generation failed - please check the source document"

def chunk_text(text, max_chars=9000):
    """Split text into chunks of max_chars length, on paragraph boundaries if possible."""
    logger.debug(f"Chunking text of length {len(text)}")
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    logger.debug(f"Text split into {len(chunks)} chunks")
    return chunks

@retry_on_failure(max_retries=2, delay=2)
def extract_for_summarization(pdf_path):
    """Extract text from PDF with error handling"""
    logger.debug(f"Extracting text from PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    formatted = ""
    
    try:
        for page_num, page in enumerate(doc):
            try:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            spans = [span["text"].strip() for span in line["spans"] if span["text"].strip()]
                            line_text = "  ".join(spans)
                            formatted += line_text + "\n"
                formatted += "\n"
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {e}")
                continue
    finally:
        doc.close()
    
    result = formatted.strip() if formatted.strip() else None
    logger.debug(f"Extracted {len(result) if result else 0} characters from PDF")
    return result

def get_fo_stocks_from_excel(filename):
    """Load F&O stocks from Excel with error handling"""
    logger.debug(f"Loading F&O stocks from {filename}")
    
    if not os.path.exists(filename):
        logger.error(f"Excel file not found: {filename}")
        return set()
    
    try:
        wb = openpyxl.load_workbook(filename)
        ws = wb.active
        fo_symbols = set()
        nse_col = None
        
        # Find the NSE Symbol column index
        for idx, cell in enumerate(ws[1], 1):
            if str(cell.value).strip().lower() == "nse symbol":
                nse_col = idx
                break
        
        if not nse_col:
            logger.warning("NSE Symbol column not found in Excel file")
            return fo_symbols
        
        for row in ws.iter_rows(min_row=2):
            cell = row[nse_col - 1]
            # Check if cell has a fill color (not white or none)
            fill = cell.fill
            fgColor = fill.fgColor
            is_colored = False
            
            if fill and fgColor:
                # Check for RGB color
                if fgColor.type == 'rgb' and fgColor.rgb not in ('00000000', 'FFFFFFFF', 'FFFFFF', None):
                    is_colored = True
                # Check for indexed color (not default)
                elif fgColor.type == 'indexed' and fgColor.indexed not in (64, 0):
                    is_colored = True
                # Check for theme color (not default)
                elif fgColor.type == 'theme':
                    is_colored = True
            
            if is_colored:
                symbol = str(cell.value).strip()
                if symbol:
                    fo_symbols.add(symbol)
        
        logger.info(f"Loaded {len(fo_symbols)} F&O symbols")
        return fo_symbols
    
    except Exception as e:
        logger.error(f"Error loading F&O stocks: {e}")
        return set()

@retry_on_failure(max_retries=3, delay=2)
def send_slack_notification(message, is_fo=False):
    """Send Slack notification with retry logic"""
    SLACK_WEBHOOK_URL1 = os.getenv("SLACK_RESULT_WEBHOOK")
    
    if not SLACK_WEBHOOK_URL1:
        logger.error("SLACK_WEBHOOK_URL1 not set in environment")
        return False
    
    logger.debug(f"Sending Slack notification (F&O: {is_fo})")
    
    if is_fo:
        payload = {
            "attachments": [
                {
                    "color": "#36a64f",
                    "text": message,
                },
            ],
            "username": "Event Bot",
            "icon_emoji": ":bell:"
        }
    else:
        payload = {
            "text": message,
            "username": "Event Bot",
            "icon_emoji": ":bell:"
        }
    
    response = requests.post(
        SLACK_WEBHOOK_URL1,
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'},
        timeout=10
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to send Slack notification: {response.status_code} - {response.text}")
        raise Exception(f"Slack notification failed: {response.status_code}")
    
    logger.debug("Slack notification sent successfully")
    return True

@retry_on_failure(max_retries=3, delay=2)
def download_pdf(url, filename):
    """Download PDF with retry logic"""
    logger.debug(f"Downloading PDF from {url}")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        f.write(response.content)
    
    logger.debug(f"PDF downloaded successfully: {filename}")

def format_market_cap(market_cap):
    """Format market cap with error handling"""
    if market_cap is None or pd.isna(market_cap):
        return ""
    
    try:
        cr = market_cap / 1e7
        if cr >= 1e5:
            return f"₹{cr/1e5:.2f}L Cr"
        elif cr >= 1e3:
            return f"₹{cr/1e3:.2f}K Cr"
        else:
            return f"₹{cr:.2f} Cr"
    except Exception as e:
        logger.warning(f"Error formatting market cap {market_cap}: {e}")
        return ""

SENT_LINKS_FILE = "sent_links.txt"

def load_sent_links():
    """Load sent links with error handling"""
    logger.debug("Loading sent links")
    
    if not os.path.exists(SENT_LINKS_FILE):
        logger.debug("Sent links file not found, creating new set")
        return set()
    
    try:
        with open(SENT_LINKS_FILE, "r") as f:
            links = set(line.strip() for line in f if line.strip())
        logger.debug(f"Loaded {len(links)} sent links")
        return links
    except Exception as e:
        logger.error(f"Error loading sent links: {e}")
        return set()

def save_sent_link(link):
    """Save sent link with error handling"""
    try:
        with open(SENT_LINKS_FILE, "a") as f:
            f.write(link + "\n")
        logger.debug(f"Saved sent link: {link}")
    except Exception as e:
        logger.error(f"Error saving sent link: {e}")

def extract_timestamp_from_text(text):
    """Extract timestamp from text with error handling"""
    try:
        # Looks for DD-MM-YYYY HH:MM:SS
        match = re.search(r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', text)
        if match:
            return match.group(1)
    except Exception as e:
        logger.warning(f"Error extracting timestamp: {e}")
    return None

def create_webdriver():
    """Create webdriver with comprehensive error handling"""
    logger.debug("Creating webdriver")
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-logging')
    options.add_argument('--log-level=3')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        logger.debug("Webdriver created successfully")
        return driver
    except Exception as e:
        logger.error(f"Error creating webdriver: {e}")
        raise

def process_bse_page(driver, wait):
    """Process BSE page and extract announcements"""
    logger.info("Processing BSE page")
    
    # Set date filters
    from_date = to_date = datetime.today().strftime("%d/%m/%Y")
    
    wait.until(EC.presence_of_element_located((By.ID, "txtFromDt")))
    wait.until(EC.presence_of_element_located((By.ID, "txtToDt")))
    
    driver.execute_script(f"document.getElementById('txtFromDt').value = '{from_date}';")
    driver.execute_script(f"document.getElementById('txtToDt').value = '{to_date}';")
    time.sleep(1)
    
    driver.execute_script("$('#txtFromDt').trigger('change');")
    driver.execute_script("$('#txtToDt').trigger('change');")

    # Select category
    category_select = wait.until(EC.presence_of_element_located((By.ID, "ddlPeriod")))
    select = Select(category_select)
    select.select_by_visible_text("Result")

    # Submit form
    submit_btn = wait.until(EC.element_to_be_clickable((By.ID, "btnSubmit")))
    driver.execute_script("arguments[0].scrollIntoView(true);", submit_btn)
    time.sleep(1)
    submit_btn.click()

    wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
    time.sleep(2)

    new_announcements = []
    from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException

    page_no = 1
    while True and not shutdown_handler.should_shutdown():
        logger.debug(f"Processing page {page_no}")
        
        for table in driver.find_elements(By.TAG_NAME, "table"):
            for a_tag in table.find_elements(By.XPATH, ".//a[contains(@href, '.pdf')]"):
                href = a_tag.get_attribute('href')
                if href:
                    try:
                        tr = a_tag.find_element(By.XPATH, "./ancestor::tr[1]")
                        first_td = tr.find_element(By.XPATH, "./td[1]")
                        name = first_td.text.strip()
                        
                        # Extract timestamp
                        timestamp_text = None
                        try:
                            parent = tr.find_element(By.XPATH, "..")
                            all_trs = parent.find_elements(By.XPATH, "./tr")
                            if len(all_trs) >= 2:
                                second_last_tr = all_trs[-2]
                                b_tags_in_second_last = second_last_tr.find_elements(By.TAG_NAME, "b")
                                if b_tags_in_second_last:
                                    timestamp_text = extract_timestamp_from_text(b_tags_in_second_last[0].text)
                        except Exception as e:
                            logger.debug(f"Error extracting timestamp: {e}")
                        
                        if not timestamp_text:
                            timestamp_text = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                        
                        new_announcements.append({
                            'Name': name,
                            'Link': href,
                            'Time': timestamp_text
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing announcement: {e}")
                        continue

        # Try to go to next page
        try:
            next_btn = driver.find_element(By.ID, "idnext")
            if next_btn.is_displayed() and next_btn.is_enabled():
                driver.execute_script("arguments[0].scrollIntoView(true);", next_btn)
                time.sleep(1)
                next_btn.click()
                page_no += 1
                time.sleep(2)
                
                # Wait for page to load
                for _ in range(10):
                    try:
                        next_btn = driver.find_element(By.ID, "idnext")
                        if next_btn.is_enabled():
                            break
                    except (StaleElementReferenceException, NoSuchElementException):
                        pass
                    time.sleep(1)
            else:
                logger.debug("No more pages to process")
                break
        except Exception as e:
            logger.debug(f"Error navigating to next page: {e}")
            break

    logger.info(f"Found {len(new_announcements)} total announcements")
    return new_announcements

def process_announcement(ann, market_cap, fo_symbols, sent_links):
    """Process individual announcement with comprehensive error handling"""
    name = ann['Name']
    link = ann['Link']
    
    if link in sent_links:
        logger.debug(f"Skipping duplicate link: {link}")
        return False
    
    logger.info(f"Processing announcement: {name}")
    
    try:
        # Extract company code
        try:
            name_parts = name.split('-')
            code = name_parts[1].strip()
        except Exception:
            code = ""
            logger.warning(f"Could not extract code from name: {name}")
        
        # Get company information
        company_name = ""
        final_market_cap = ""
        nse_symbol = None
        raw_market_cap = None
        industry = None
        
        if code:
            mc_row = market_cap[market_cap['BSE Code'] == code]
            if not mc_row.empty:
                company_name = mc_row.iloc[0]['Company Name']
                nse_symbol = mc_row.iloc[0]['NSE Symbol']
                industry = mc_row.iloc[0]['Industry']
                excel_market_cap = mc_row.iloc[0].get('Latest Market Cap', None)
                
                # Try to get market cap from Yahoo Finance
                market_cap_yf = None
                if pd.notnull(nse_symbol):
                    ticker_symbol = f"{nse_symbol}.NS"
                    try:
                        ticker = yf.Ticker(ticker_symbol)
                        info = ticker.info
                        market_cap_yf = info.get('marketCap', None)
                    except Exception as e:
                        logger.debug(f"Error fetching Yahoo Finance data for {ticker_symbol}: {e}")
                
                # Determine final market cap
                if market_cap_yf and market_cap_yf > 0:
                    raw_market_cap = market_cap_yf
                elif excel_market_cap and excel_market_cap > 0:
                    raw_market_cap = excel_market_cap
        
        # Check market cap threshold
        if raw_market_cap is None or raw_market_cap < 1000 * 1e7:
            logger.info(f"Skipping due to low market cap: {raw_market_cap} for {name}")
            return False
        
        final_market_cap = format_market_cap(raw_market_cap)
        summary_name = company_name if company_name else name
        
        # Process PDF
        pdf_filename = f"temp_{int(time.time())}_{os.getpid()}.pdf"
        summary = "Error processing PDF"
        
        try:
            download_pdf(link, pdf_filename)
            text = extract_for_summarization(pdf_filename)
            
            if text:
                # Chunk if too large
                chunks = chunk_text(text, max_chars=9000)
                if len(chunks) == 1:
                    summary = summarise_bse_text(chunks[0])
                else:
                    logger.info(f"Text too long, splitting into {len(chunks)} chunks")
                    chunk_summaries = []
                    for j, chunk in enumerate(chunks):
                        logger.debug(f"Summarizing chunk {j+1}/{len(chunks)}")
                        chunk_summary = summarise_bse_text(chunk)
                        chunk_summaries.append(chunk_summary)
                        time.sleep(2)  # Rate limiting
                    
                    merged_summary_text = "\n".join(chunk_summaries)
                    logger.debug("Summarizing merged chunk summaries")
                    summary = summarise_bse_text(merged_summary_text)
            else:
                summary = "No text extracted from PDF"
                
        except Exception as e:
            logger.error(f"Error processing PDF for {name}: {e}")
            summary = f"Error processing document: {str(e)}"
        finally:
            # Clean up PDF file
            try:
                if os.path.exists(pdf_filename):
                    os.remove(pdf_filename)
            except Exception as e:
                logger.warning(f"Error removing PDF file: {e}")
        
        # Send notification
        release_time = ann.get('Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        slack_message = (
            f"*{summary_name}* ({final_market_cap})\n"
            f"*Summary:* {summary}\n"
            f"*Industry:* {industry}\n"
            f"*Link:* {link}\n"
            f"*Release Time:* {release_time}\n"
        )
        
        is_fo = nse_symbol and str(nse_symbol).strip() in fo_symbols
        
        if send_slack_notification(slack_message, is_fo=is_fo):
            sent_links.add(link)
            save_sent_link(link)
            logger.info(f"Successfully processed and sent notification for: {summary_name}")
            health_monitor.record_success()
            return True
        else:
            logger.error(f"Failed to send notification for: {summary_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing announcement {name}: {e}")
        logger.error(traceback.format_exc())
        health_monitor.record_failure()
        return False

def log_health_status():
    """Log current health status"""
    status = health_monitor.get_health_status()
    logger.info(f"Health Status: {json.dumps(status, indent=2)}")

def main():
    """Main function with comprehensive error handling"""
    logger.info("Starting BSE Scraper with enhanced logging")
    
    # Load environment variables
    load_dotenv()
    
    # Validate required environment variables
    required_vars = ["OPENROUTER_API_KEY", "SLACK_RESULT_WEBHOOK"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    # Load market cap data
    try:
        market_cap = pd.read_excel("Market Cap.xlsx")
        market_cap['BSE Code'] = (
            market_cap['BSE Code']
            .astype(str)
            .str.replace('.0', '', regex=False)
            .str.strip()
        )
        logger.info(f"Loaded market cap data with {len(market_cap)} entries")
    except Exception as e:
        logger.error(f"Error loading market cap data: {e}")
        sys.exit(1)
    
    # Load F&O symbols
    fo_symbols = get_fo_stocks_from_excel("Market Cap.xlsx")
    
    # Load sent links
    sent_links = load_sent_links()
    
    # Start health monitoring thread
    def health_monitor_thread():
        while not shutdown_handler.should_shutdown():
            try:
                log_health_status()
                time.sleep(300)  # Log every 5 minutes
            except Exception as e:
                logger.error(f"Error in health monitor thread: {e}")
                time.sleep(60)
    
    health_thread = threading.Thread(target=health_monitor_thread, daemon=True)
    health_thread.start()
    
    # Main processing loop
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while not shutdown_handler.should_shutdown():
        try:
            logger.info(f"[{datetime.now()}] Starting new BSE check cycle")
            
            # Create webdriver
            driver = None
            try:
                driver = create_webdriver()
                driver.get("https://www.bseindia.com/corporates/ann.html")
                wait = WebDriverWait(driver, 20)
                
                # Process BSE page
                new_announcements = process_bse_page(driver, wait)
                
                # Filter out already sent links
                filtered_announcements = [
                    ann for ann in new_announcements 
                    if ann['Link'] not in sent_links
                ]
                
                logger.info(f"Found {len(filtered_announcements)} new announcements to process")
                
                # Process announcements in batches
                batch_size = 10
                for i in range(0, len(filtered_announcements), batch_size):
                    if shutdown_handler.should_shutdown():
                        break
                        
                    batch = filtered_announcements[i:i+batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} announcements)")
                    
                    for ann in batch:
                        if shutdown_handler.should_shutdown():
                            break
                            
                        try:
                            success = process_announcement(ann, market_cap, fo_symbols, sent_links)
                            if success:
                                # Send separator
                                send_slack_notification("-----\n\n")
                                time.sleep(2)  # Rate limiting
                        except Exception as e:
                            logger.error(f"Error processing individual announcement: {e}")
                            health_monitor.record_failure()
                            continue
                    
                    # Wait between batches (except for last batch)
                    if i + batch_size < len(filtered_announcements) and not shutdown_handler.should_shutdown():
                        logger.info("Waiting 60 seconds before next batch...")
                        time.sleep(60)
                
                # Reset consecutive failures on successful cycle
                consecutive_failures = 0
                logger.info("BSE check cycle completed successfully")
                
            except Exception as e:
                logger.error(f"Error in BSE processing cycle: {e}")
                logger.error(traceback.format_exc())
                consecutive_failures += 1
                health_monitor.record_failure()
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(f"Too many consecutive failures ({consecutive_failures}). Implementing extended backoff.")
                    time.sleep(300)  # 5 minute extended backoff
                    consecutive_failures = 0
                
            finally:
                # Clean up webdriver
                if driver:
                    try:
                        driver.quit()
                        logger.debug("Webdriver closed successfully")
                    except Exception as e:
                        logger.warning(f"Error closing webdriver: {e}")
            
            # Wait before next cycle
            if not shutdown_handler.should_shutdown():
                sleep_time = 60 if consecutive_failures == 0 else min(60 * (2 ** consecutive_failures), 300)
                logger.info(f"Waiting {sleep_time} seconds before next cycle...")
                
                # Interruptible sleep
                for _ in range(sleep_time):
                    if shutdown_handler.should_shutdown():
                        break
                    time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            logger.error(traceback.format_exc())
            consecutive_failures += 1
            health_monitor.record_failure()
            
            # Emergency sleep
            time.sleep(30)
    
    logger.info("BSE Scraper shutting down gracefully")
    log_health_status()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)