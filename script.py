import os
import time
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
from openai import OpenAI
import fitz
from groq import Groq
import requests
import logging

# --- Logging Setup ---
LOG_FILE = "bse_scraper_script.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set your API keys (or use os.getenv for .env integration)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

PROMPT_TEMPLATE = (
    "You are a financial analyst. Read the following BSE company update and write a one-line summary focusing only on the most important detail. "
    "Do not include any introductory or closing phrases, and do not write anything except the summary itself.\n\n{text}\n"
)

def summarise_with_groq_model(text, model, stream=False, max_retries=3):
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set")
        return None
    client = Groq(api_key=GROQ_API_KEY)
    prompt = PROMPT_TEMPLATE.format(text=text)
    for attempt in range(max_retries):
        try:
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
        except Exception as e:
            logger.warning(f"[{model}] Attempt {attempt + 1} failed: {e}")
            time.sleep(3 * (attempt + 1))
    return None

def chunk_text(text, max_chars=9000):
    """Split text into chunks of max_chars length, on paragraph boundaries if possible."""
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
    return chunks

def summarise_with_gemini(text, max_retries=3):
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    prompt = PROMPT_TEMPLATE.format(text=text)
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 429:
                logger.warning("[Gemini] Rate limit hit. Retrying...")
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"[Gemini] Attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return None

def summarise_with_openrouter(text, api_key=OPENROUTER_API_KEY, max_retries=5):
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return None
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    prompt = PROMPT_TEMPLATE.format(text=text)
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="deepseek/deepseek-r1-0528:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"[OpenRouter] Attempt {attempt + 1} failed: {e}")
            time.sleep(10 * (attempt + 1))
    return None

def summarise_bse_text(text, stream=False):
    groq_models = [
        "llama3-8b-8192",
        "deepseek-r1-distill-llama-70b",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
    for model in groq_models:
        logger.info(f"Trying {model}...")
        summary = summarise_with_groq_model(text, model, stream=stream)
        if summary:
            logger.info(f"✔ Success with {model}")
            return summary
        logger.info(f"✘ Failed with {model}, trying next...")

    logger.info("⚠ Falling back to Gemini...")
    summary = summarise_with_gemini(text)
    if summary:
        logger.info("✔ Success with Gemini")
        return summary

    logger.info("⚠ Falling back to OpenRouter...")
    summary = summarise_with_openrouter(text)
    if summary:
        logger.info("✔ Success with OpenRouter")
        return summary

    logger.error("All summarization attempts failed.")
    return "Summary generation failed."

# --- PDF extraction ---
def extract_for_summarization(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        formatted = ""
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        spans = [span["text"].strip() for span in line["spans"] if span["text"].strip()]
                        line_text = "  ".join(spans)
                        formatted += line_text + "\n"
            formatted += "\n"
        doc.close()
        return formatted.strip() if formatted.strip() else None
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return None

# --- Download PDF ---
def download_pdf(url, filename):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        logger.error(f"Error downloading PDF from {url}: {e}")
        raise

# --- Market Cap Formatting ---
def format_market_cap(market_cap):
    if market_cap is None or pd.isna(market_cap):
        return ""
    return f"₹{market_cap/1e7:.2f} Cr"

# --- Main Loop ---
def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")  # Use OpenRouter API Key
    if not api_key:
        logger.error("OpenRouter API Key not found in .env file.")
        return

    # Load market cap data
    try:
        market_cap = pd.read_excel("Market Cap.xlsx")
        market_cap['BSE Code'] = (
            market_cap['BSE Code']
            .astype(str)
            .str.replace('.0', '', regex=False)
            .str.strip()
        )
    except Exception as e:
        logger.error(f"Error loading Market Cap.xlsx: {e}")
        return

    # Excel output file
    today_str = datetime.today().strftime('%Y-%m-%d')
    summary_file = f"news_{today_str}.xlsx"

    # Load previous summaries
    if os.path.exists(summary_file):
        try:
            summary_df = pd.read_excel(summary_file)
            existing_links = set(summary_df['Link'])
        except Exception as e:
            logger.error(f"Error loading {summary_file}: {e}")
            summary_df = pd.DataFrame(columns=['Company Name', 'Market Cap', 'Summary', 'Link'])
            existing_links = set()
    else:
        summary_df = pd.DataFrame(columns=['Company Name', 'Market Cap', 'Summary', 'Link'])
        existing_links = set()

    # Selenium setup
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(ChromeDriverManager().install())

    while True:
        try:
            logger.info(f"[{datetime.now()}] Checking for new BSE announcements...")

            driver = webdriver.Chrome(service=service, options=options)
            driver.get("https://www.bseindia.com/corporates/ann.html")
            wait = WebDriverWait(driver, 20)

            # Set today's date
            from_date = to_date = "04/07/2025" #datetime.today().strftime("%d/%m/%Y")
            wait.until(EC.presence_of_element_located((By.ID, "txtFromDt")))
            wait.until(EC.presence_of_element_located((By.ID, "txtToDt")))
            driver.execute_script(f"document.getElementById('txtFromDt').value = '{from_date}';")
            driver.execute_script(f"document.getElementById('txtToDt').value = '{to_date}';")
            time.sleep(1)
            driver.execute_script("$('#txtFromDt').trigger('change');")
            driver.execute_script("$('#txtToDt').trigger('change');")

            category_select = wait.until(EC.presence_of_element_located((By.ID, "ddlPeriod")))
            select = Select(category_select)
            select.select_by_visible_text("Company Update")

            submit_btn = wait.until(EC.element_to_be_clickable((By.ID, "btnSubmit")))
            driver.execute_script("arguments[0].scrollIntoView(true);", submit_btn)
            time.sleep(1)
            submit_btn.click()

            wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            time.sleep(2)

            # Scrape all announcements with pagination
            new_announcements = []
            from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException

            page_no = 1
            while True:
                for table in driver.find_elements(By.TAG_NAME, "table"):
                    for a_tag in table.find_elements(By.XPATH, ".//a[contains(@href, '.pdf')]"):
                        href = a_tag.get_attribute('href')
                        if href and href not in existing_links:
                            tr = a_tag.find_element(By.XPATH, "./ancestor::tr[1]")
                            first_td = tr.find_element(By.XPATH, "./td[1]")
                            name = first_td.text.strip()
                            new_announcements.append({'Name': name, 'Link': href})

                try:
                    next_btn = driver.find_element(By.ID, "idnext")
                    if next_btn.is_displayed() and next_btn.is_enabled():
                        driver.execute_script("arguments[0].scrollIntoView(true);", next_btn)
                        time.sleep(1)
                        next_btn.click()
                        page_no += 1
                        logger.info(f"Navigating to page {page_no}...")
                        time.sleep(2)
                        # Optionally, wait for the next button to become enabled again
                        for _ in range(10):
                            try:
                                next_btn = driver.find_element(By.ID, "idnext")
                                if next_btn.is_enabled():
                                    break
                            except (StaleElementReferenceException, NoSuchElementException):
                                pass
                            time.sleep(1)
                    else:
                        break
                except Exception as e:
                    logger.warning(f"Error navigating to next page: {e}")
                    break

            driver.quit()

            logger.info(f"Found {len(new_announcements)} new announcements.")

            # Summarize new announcements
            for ann in new_announcements:
                name = ann['Name']
                link = ann['Link']
                try:
                    name_parts = name.split('-')
                    code = name_parts[1].strip()
                    additional_info = name_parts[2].strip() if len(name_parts) > 2 else ""
                except Exception:
                    code = ""
                    additional_info = ""
                company_name = ""
                final_market_cap = ""

                mc_row = market_cap[market_cap['BSE Code'] == code]
                nse_symbol = None
                industry = None
                if not mc_row.empty:
                    company_name = mc_row.iloc[0]['Company Name']
                    nse_symbol = mc_row.iloc[0]['NSE Symbol']
                    industry = mc_row.iloc[0]['Industry'] if 'Industry' in mc_row.columns else None
                    excel_market_cap = mc_row.iloc[0].get('Latest Market Cap', None)
                    market_cap_yf = None
                    if pd.notnull(nse_symbol):
                        ticker_symbol = f"{nse_symbol}.NS"
                        try:
                            ticker = yf.Ticker(ticker_symbol)
                            info = ticker.info
                            market_cap_yf = info.get('marketCap', None)
                        except Exception as e:
                            logger.warning(f"Yahoo Finance error for {ticker_symbol}: {e}")
                        if market_cap_yf and market_cap_yf > 0:
                            final_market_cap = format_market_cap(market_cap_yf)
                        elif excel_market_cap and excel_market_cap > 0:
                            final_market_cap = format_market_cap(excel_market_cap)
                    elif excel_market_cap and excel_market_cap > 0:
                        final_market_cap = format_market_cap(excel_market_cap)

                summary_name = company_name if company_name else name

                pdf_filename = f"temp_{int(time.time())}.pdf"
                try:
                    download_pdf(link, pdf_filename)
                    text = extract_for_summarization(pdf_filename)
                    if text:
                        # Chunk if too large
                        chunks = chunk_text(text, max_chars=9000)
                        if len(chunks) == 1:
                            summary = summarise_bse_text(chunks[0])
                        else:
                            logger.info(f"Text too long, splitting into {len(chunks)} chunks.")
                            chunk_summaries = []
                            for i, chunk in enumerate(chunks):
                                logger.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
                                chunk_summary = summarise_bse_text(chunk)
                                chunk_summaries.append(chunk_summary)
                                time.sleep(5)
                            merged_summary_text = "\n".join(chunk_summaries)
                            logger.info("Summarizing merged chunk summaries...")
                            summary = summarise_bse_text(merged_summary_text)
                    else:
                        summary = "No text extracted from PDF."
                except Exception as e:
                    summary = f"Error processing {name}: {str(e)}"
                    logger.error(summary)
                finally:
                    if os.path.exists(pdf_filename):
                        try:
                            os.remove(pdf_filename)
                        except Exception as e:
                            logger.warning(f"Could not remove temp file {pdf_filename}: {e}")

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                summary_df = pd.concat([
                    summary_df,
                    pd.DataFrame([{
                        'Company Name': summary_name,
                        'Market Cap': final_market_cap,
                        'Summary': summary,
                        'Industry': industry,
                        'Link': link,
                        'Timestamp': timestamp,
                        'Additional Info': additional_info,
                    }])
                ], ignore_index=True)
                existing_links.add(link)
                logger.info(f"Added summary for: {summary_name}")

                summary_df = summary_df.drop_duplicates(subset=['Link'])
                try:
                    summary_df.to_excel(summary_file, index=False)
                    logger.info(f"Saved {len(summary_df)} summaries to {summary_file}")
                except Exception as e:
                    logger.error(f"Error saving to {summary_file}: {e}")

            # Wait 1 minute before next check
            time.sleep(60)
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()