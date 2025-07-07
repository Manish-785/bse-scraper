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

# --- Summarization Backends ---
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

PROMPT_TEMPLATE = (
    "You are a financial analyst. Read the following BSE company update and write a one-line summary focusing only on the most important detail. "
    "Do not include any introductory or closing phrases, and do not write anything except the summary itself.\n\n{text}\n"
)

def summarise_with_groq_model(text, model, stream=False, max_retries=3):
    if not GROQ_API_KEY:
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
            print(f"[{model}] Attempt {attempt + 1} failed: {e}")
            time.sleep(3 * (attempt + 1))
    return None

def summarise_with_gemini(text, max_retries=3):
    if not GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    prompt = PROMPT_TEMPLATE.format(text=text)
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 429:
                print("[Gemini] Rate limit hit. Retrying...")
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"[Gemini] Attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return None

def summarise_with_openrouter(text, api_key=OPENROUTER_API_KEY, max_retries=5):
    if not api_key:
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
            print(f"[OpenRouter] Attempt {attempt + 1} failed: {e}")
            time.sleep(10 * (attempt + 1))
    return None

def summarise_bse_text(text, stream=False):
    groq_models = [
        "llama3-70b-8192",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant",
    ]
    for model in groq_models:
        print(f"Trying {model}...")
        summary = summarise_with_groq_model(text, model, stream=stream)
        if summary:
            print(f"✔ Success with {model}")
            return summary
        print(f"✘ Failed with {model}, trying next...")

    print("⚠ Falling back to Gemini...")
    summary = summarise_with_gemini(text)
    if summary:
        print("✔ Success with Gemini")
        return summary

    print("⚠ Falling back to OpenRouter...")
    summary = summarise_with_openrouter(text)
    if summary:
        print("✔ Success with OpenRouter")
        return summary

    raise Exception("All summarization attempts failed.")

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

def extract_for_summarization(pdf_path):
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

def get_fo_stocks_from_excel(filename):
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
    return fo_symbols

def send_slack_notification(message, is_fo=False):
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    if not SLACK_WEBHOOK_URL:
        print("SLACK_WEBHOOK_URL not set in environment.")
        return
    
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
        SLACK_WEBHOOK_URL,
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code != 200:
        print(f"Failed to send Slack notification: {response.text}")

def download_pdf(url, filename):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, allow_redirects=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)

def format_market_cap(market_cap):
    if market_cap is None or pd.isna(market_cap):
        return ""
    cr = market_cap / 1e7
    if cr >= 1e5:
        return f"₹{cr/1e5:.2f}L Cr"
    elif cr >= 1e3:
        return f"₹{cr/1e3:.2f}K Cr"
    else:
        return f"₹{cr:.2f} Cr"

SENT_LINKS_FILE = "sent_links.txt"

def load_sent_links():
    if not os.path.exists(SENT_LINKS_FILE):
        return set()
    with open(SENT_LINKS_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_sent_link(link):
    with open(SENT_LINKS_FILE, "a") as f:
        f.write(link + "\n")

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OpenRouter API Key not found in .env file.")
        return

    market_cap = pd.read_excel("Market Cap.xlsx")
    market_cap['BSE Code'] = (
        market_cap['BSE Code']
        .astype(str)
        .str.replace('.0', '', regex=False)
        .str.strip()
    )
    fo_symbols = get_fo_stocks_from_excel("Market Cap.xlsx")
    
    sent_links = load_sent_links()

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(ChromeDriverManager().install())

    while True:
        print(f"\n[{datetime.now()}] Checking for new BSE announcements...")

        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://www.bseindia.com/corporates/ann.html")
        wait = WebDriverWait(driver, 20)

        from_date = to_date = datetime.today().strftime("%d/%m/%Y")
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

        new_announcements = []
        from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException

        page_no = 1
        while True:
            for table in driver.find_elements(By.TAG_NAME, "table"):
                for a_tag in table.find_elements(By.XPATH, ".//a[contains(@href, '.pdf')]"):
                    href = a_tag.get_attribute('href')
                    if href and href not in sent_links:
                        tr = a_tag.find_element(By.XPATH, "./ancestor::tr[1]")
                        tds = tr.find_elements(By.TAG_NAME, "td")
                        first_td = tr.find_element(By.XPATH, "./td[1]")
                        name = first_td.text.strip()
                        try:
                            b_tags = tr.find_elements(By.XPATH, ".//b[@class='ng-binding']")
                            timestamp_text = b_tags[0].text.strip() if b_tags else datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                        except:
                            timestamp_text = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                        new_announcements.append({'Name': name, 'Link': href, 'Time': timestamp_text})

            try:
                next_btn = driver.find_element(By.ID, "idnext")
                if next_btn.is_displayed() and next_btn.is_enabled():
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_btn)
                    time.sleep(1)
                    next_btn.click()
                    page_no += 1
                    print(f"Navigating to page {page_no}...")
                    time.sleep(2)
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
                print(f"Error navigating to next page: {e}")
                break

        driver.quit()

        print(f"Found {len(new_announcements)} new announcements.")

        # Process and send announcements in batches of 10 per minute
        for i in range(0, len(new_announcements), 10):
            batch = new_announcements[i:i+10]
            for ann in batch:
                name = ann['Name']
                link = ann['Link']
                
                if "newspaper publication" in name.lower():
                    print(f"Skipping newspaper publication: {name}")
                    continue
                if link in sent_links:
                    print(f"Skipping duplicate link: {link}")
                    continue
                try:
                    name_parts = name.split('-')
                    code = name_parts[1].strip()
                except Exception:
                    code = ""
                company_name = ""
                final_market_cap = ""

                mc_row = market_cap[market_cap['BSE Code'] == code]
                nse_symbol = None
                raw_market_cap = None
                industry = None
                if not mc_row.empty:
                    company_name = mc_row.iloc[0]['Company Name']
                    nse_symbol = mc_row.iloc[0]['NSE Symbol']
                    industry = mc_row.iloc[0]['Industry']
                    excel_market_cap = mc_row.iloc[0].get('Latest Market Cap', None)
                    market_cap_yf = None
                    if pd.notnull(nse_symbol):
                        ticker_symbol = f"{nse_symbol}.NS"
                        ticker = yf.Ticker(ticker_symbol)
                        try:
                            info = ticker.info
                        except Exception as e:
                            print(f"Error fetching data for {ticker_symbol}: {e}")
                            info = {}
                        market_cap_yf = info.get('marketCap', None)
                        if market_cap_yf and market_cap_yf > 0:
                            # final_market_cap = format_market_cap(market_cap_yf)
                            raw_market_cap = market_cap_yf
                        elif excel_market_cap and excel_market_cap > 0:
                            # final_market_cap = format_market_cap(excel_market_cap)
                            raw_market_cap = excel_market_cap
                    elif excel_market_cap and excel_market_cap > 0:
                        # final_market_cap = format_market_cap(excel_market_cap)
                        raw_market_cap = excel_market_cap
                        

                try:
                    if (raw_market_cap is None) or (raw_market_cap < 500 * 1e7):
                        print(f"Skipping due to low market cap: {raw_market_cap} for {name}")
                        continue
                    final_market_cap = format_market_cap(raw_market_cap)
                except Exception:
                    print(f"Skipping due to invalid market cap: {raw_market_cap} for {name}")
                    continue
                
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
                            print(f"Text too long, splitting into {len(chunks)} chunks.")
                            chunk_summaries = []
                            for j, chunk in enumerate(chunks):
                                print(f"Summarizing chunk {j+1}/{len(chunks)}...")
                                chunk_summary = summarise_bse_text(chunk)
                                chunk_summaries.append(chunk_summary)
                                time.sleep(5)
                            merged_summary_text = "\n".join(chunk_summaries)
                            print("Summarizing merged chunk summaries...")
                            summary = summarise_bse_text(merged_summary_text)
                    else:
                        summary = "No text extracted from PDF."
                except Exception as e:
                    summary = f"Error processing {name}: {str(e)}"
                finally:
                    if os.path.exists(pdf_filename):
                        os.remove(pdf_filename)

                release_time = ann.get('Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                slack_message = (
                    f"*{summary_name}* ({final_market_cap})\n"
                    f"*Summary:* {summary}\n"
                    f"*Industry:* {industry}"
                    f"*Link:* {link}\n"
                    f"*Release Time:* {release_time}\n"
                )
                is_fo = True if nse_symbol and str(nse_symbol).strip() in fo_symbols else False
                send_slack_notification(slack_message,is_fo=is_fo)
                sent_links.add(link)
                save_sent_link(link)
                print(f"Sent Slack notification for: {summary_name}")
                
                send_slack_notification("-----\n\n")

            if i + 10 < len(new_announcements):
                print("Sent 10 announcements. Waiting 60 seconds before next batch...")
                time.sleep(60)

        print("Sleeping for 60 seconds before next page refresh...")
        time.sleep(60)

if __name__ == "__main__":
    main()