"""
SEC EDGAR Crawler — Supply Chain Network Extraction via NLP
Module: SEC EDGAR Crawler
Owner: Nguyễn Vũ Thủy
 HUST - Group 7
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Constants and Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "SupplyChainNLP/1.0 group7@hust.edu.vn")

# SEC EDGAR requires a specific, professional User-Agent header
HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate"
}

# Known name overrides for SEC CIK resolution lookup mapping
NAME_OVERRIDES = {
    "Samsung Electronics Co., Ltd.": "Samsung Electronics",
    "Sony Group Corporation": "Sony Group",
    "Taiwan Semiconductor Manufacturing Company Limited": "Taiwan Semiconductor",
    "ASML Holding N.V.": "ASML",
    "Advanced Micro Devices, Inc.": "Advanced Micro Devices",
    "Dell Technologies Inc.": "Dell Technologies",
    "HP Inc.": "HP",
    "Broadcom Inc.": "Broadcom",
    "Qualcomm Incorporated": "Qualcomm",
    "Texas Instruments Incorporated": "Texas Instruments",
    "Micron Technology, Inc.": "Micron Technology",
    "NVIDIA Corporation": "NVIDIA",
    "Intel Corporation": "Intel",
    "Microsoft Corporation": "Microsoft",
    "Apple Inc.": "Apple"
}

# Supply-chain-relevant keywords (stem matching)
SUPPLY_KEYWORDS = [
    "suppli", "manufactur", "vendor", "contract", "procure",
    "outsourc", "distribut", "partner", "acqui", "subsidiar",
    "joint venture", "licens", "assembly", "component", "source",
    "tier-1", "tier-2", "oem", "odm", "wafer", "foundry"
]

# Lazy-loaded spaCy model
_nlp_model = None


def get_nlp():
    """Lazy loads and returns the spaCy NLP model."""
    global _nlp_model
    if _nlp_model is None:
        import spacy
        # Load small English pipeline for sentence segmentation
        _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model


def execute_sec_request(url: str) -> requests.Response:
    """
    Executes an HTTP GET request to the SEC EDGAR API with polite rate limiting and retry logic.
    
    Rate limits: Strict maximum of 10 requests per second (time.sleep(0.11)).
    Retry on HTTP 429: Exponential back-off (5s -> 15s -> 60s), up to 3 retries.
    """
    # Adhere strictly to the SEC rate limit (≤ 10 requests / second)
    time.sleep(0.11)
    
    retries = 3
    backoff_seconds = [5, 15, 60]
    
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            
            # Handle rate limiting from SEC EDGAR
            if response.status_code == 429:
                if attempt < retries:
                    sleep_time = backoff_seconds[attempt]
                    print(f"    [SEC 429] Rate limit hit. Retrying in {sleep_time}s... (Attempt {attempt + 1}/{retries})")
                    time.sleep(sleep_time)
                    continue
                else:
                    response.raise_for_status()
            
            response.raise_for_status()
            return response
            
        except requests.RequestException as e:
            if attempt < retries:
                sleep_time = backoff_seconds[attempt]
                print(f"    [HTTP Error] {e}. Retrying in {sleep_time}s... (Attempt {attempt + 1}/{retries})")
                time.sleep(sleep_time)
            else:
                raise


def get_cik(company_name: str) -> str | None:
    """
    Resolves a company's legal name to its SEC Central Index Key (CIK).
    
    1. Checks the NAME_OVERRIDES dictionary first.
    2. Primary method: Calls company_tickers.json endpoint.
    3. Fallback method: Queries the EFTS full-text search index if primary lookup fails.
    """
    # 1. Check overrides dictionary first
    search_name = NAME_OVERRIDES.get(company_name, company_name)
    
    # 2. Primary Method: SEC Company Tickers JSON
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        response = execute_sec_request(url)
        tickers_data = response.json()
        
        # Search for company name exact or fuzzy/cleaned case-insensitive
        target_name_clean = search_name.lower().strip()
        
        # Try exact case-insensitive match on official SEC title
        for key, entry in tickers_data.items():
            title = entry.get("title", "")
            if title.lower().strip() == target_name_clean:
                cik_str = str(entry.get("cik_str"))
                return cik_str.zfill(10)
                
        # Try partial match/substring if exact fails
        for key, entry in tickers_data.items():
            title = entry.get("title", "").lower()
            if target_name_clean in title or title in target_name_clean:
                cik_str = str(entry.get("cik_str"))
                return cik_str.zfill(10)
                
    except Exception as e:
        print(f"  [CIK Lookup] Primary lookup error: {e}. Trying fallback method...")
        
    # 3. Fallback Method: EFTS Full-text Search Index
    try:
        # Quote query to seek exact matches for name
        url = f'https://efts.sec.gov/LATEST/search-index?q="{search_name}"&forms=10-K,20-F'
        response = execute_sec_request(url)
        search_data = response.json()
        
        hits = search_data.get("hits", {}).get("hits", [])
        if hits:
            # Extract CIK from the first match
            cik_str = hits[0].get("_source", {}).get("cik")
            if cik_str:
                return str(cik_str).zfill(10)
                
    except Exception as e:
        print(f"  [CIK Lookup] Fallback lookup error: {e}")
        
    return None


def get_filings(cik: str, form_types: tuple = ("10-K", "20-F", "10-Q"), max_filings: int = 10) -> list[dict]:
    """
    Fetches the list of recent filings for a given CIK from SEC Submissions API.
    
    Returns a list of dicts: [{"accession": str, "form": str, "date": str}]
    Dashes are stripped from the accession numbers in the returned values.
    """
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    response = execute_sec_request(url)
    data = response.json()
    
    recent_filings = data.get("filings", {}).get("recent", {})
    if not recent_filings:
        return []
        
    accessions = recent_filings.get("accessionNumber", [])
    forms = recent_filings.get("form", [])
    dates = recent_filings.get("filingDate", [])
    
    filings = []
    # Walk through filings (they are ordered newest first in the submissions JSON)
    for i in range(len(forms)):
        form = forms[i]
        if form in form_types:
            raw_accession = accessions[i]
            # Strip dashes from accession number
            accession_clean = raw_accession.replace("-", "")
            filings.append({
                "accession": accession_clean,
                "form": form,
                "date": dates[i]
            })
            if len(filings) >= max_filings:
                break
                
    return filings


def fetch_filing_text(cik: str, accession: str) -> str:
    """
    Downloads the primary document from a filing and returns its plain text.
    
    1. Fetches the index page using accession dashed.
    2. Parses the HTML to find the first .htm/.html/.txt primary document link.
    3. Fetches the primary document itself.
    4. Strips HTML tags and script/style contents using BeautifulSoup, returning clean plain text.
    """
    # Accession number is 18 digits. Insert dashes to reconstruct original accession
    if len(accession) != 18:
        raise ValueError(f"Invalid accession number length ({len(accession)} chars). Expected 18.")
        
    accession_dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    cik_stripped = cik.lstrip("0")  # SEC raw data directories use CIK without leading zeros
    
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{accession}/{accession_dashed}-index.htm"
    
    # Step 1 & 2: Fetch and parse index page
    index_response = execute_sec_request(index_url)
    index_soup = BeautifulSoup(index_response.text, "lxml")
    
    primary_doc_href = None
    # Find primary document table row links
    for a in index_soup.find_all("a", href=True):
        href = a["href"].strip()
        # Avoid matching generic SEC navigation header links (e.g. /index.htm)
        if href.startswith("/") and not href.startswith("/Archives/edgar/data/"):
            continue
        # Find first link ending in target document types, avoiding index file itself
        if href.endswith((".htm", ".html", ".txt")) and not href.endswith(("-index.htm", "-index.html")):
            primary_doc_href = href
            break
            
    if not primary_doc_href:
        raise ValueError(f"Primary document link (.htm/.html/.txt) not found in index for CIK {cik}, accession {accession}")
        
    # Resolve relative URL path if necessary
    if primary_doc_href.startswith("/"):
        doc_url = f"https://www.sec.gov{primary_doc_href}"
    elif primary_doc_href.startswith("http"):
        doc_url = primary_doc_href
    else:
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{accession}/{primary_doc_href}"
        
    # Step 3: Fetch document
    doc_response = execute_sec_request(doc_url)
    
    # Step 4: Parse text and strip HTML
    doc_soup = BeautifulSoup(doc_response.text, "lxml")
    
    # Remove script and style elements entirely
    for element in doc_soup(["script", "style"]):
        element.extract()
        
    # Retrieve clean text, using single spaces as separators
    plain_text = doc_soup.get_text(separator=" ")
    return plain_text


def extract_sentences(text: str) -> list[str]:
    """
    Tokenises text into sentences and keeps only supply-chain-relevant ones.
    
    Constraints:
    - Input text is split into chunks of 500,000 characters to prevent spaCy's 1,000,000
      character memory error, ensuring 100% of the document is parsed.
    - Sentences must be between 40 and 1,000 characters long (inclusive).
    - Sentences must contain at least one supply chain keyword (stem matching).
    - Returns a deduplicated list keeping original insertion order.
    """
    if not text:
        return []
        
    # Split text into chunks of 500,000 chars to avoid memory/truncation loss
    chunk_size = 500000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    nlp = get_nlp()
    matched_sentences = []
    
    for chunk in chunks:
        if not chunk.strip():
            continue
        doc = nlp(chunk)
        for sent in doc.sents:
            # Clean whitespace and strip leading/trailing spaces
            sentence_clean = " ".join(sent.text.strip().split())
            
            # Length check
            if 40 <= len(sentence_clean) <= 1000:
                # Keyword matching (stem matching, case-insensitive substring)
                sentence_lower = sentence_clean.lower()
                if any(kw in sentence_lower for kw in SUPPLY_KEYWORDS):
                    matched_sentences.append(sentence_clean)
                
    # Deduplicate while preserving order
    return list(dict.fromkeys(matched_sentences))


def save_sentences(supabase: Client, company_name: str, cik: str, accession: str, form_type: str, filing_date: str, sentences: list[str]) -> int:
    """
    Batch-inserts sentences into the Supabase `raw_sentences` table.
    
    Chunks the input sentences into batches of 500 rows maximum.
    Uses upsert to handle unique constraints elegantly and prevent error blocks on duplicates.
    """
    if not sentences:
        return 0
        
    rows = []
    for s in sentences:
        rows.append({
            "company_name": company_name,
            "cik": cik,
            "accession_number": accession,
            "form_type": form_type,
            "filing_date": filing_date,
            "sentence": s,
            "llm_processed": False
        })
        
    total_saved = 0
    # Process batch in chunks of 500
    for i in range(0, len(rows), 500):
        batch = rows[i:i + 500]
        # Use upsert with unique index columns to prevent double inserts and handle conflicts
        response = supabase.table("raw_sentences").upsert(
            batch,
            on_conflict="company_name,accession_number,sentence"
        ).execute()
        
        # Accumulate successfully saved count
        if hasattr(response, "data") and response.data:
            total_saved += len(response.data)
        else:
            total_saved += len(batch)
            
    return total_saved


def mark_status(supabase: Client, company_name: str, status: str) -> None:
    """Updates the status in the `company_queue` table for a given company name."""
    supabase.table("company_queue").update({
        "status": status
    }).eq("company_name", company_name).execute()


def run_crawler(supabase: Client) -> None:
    """
    Main orchestration loop for the SEC EDGAR Crawler.
    Processes companies from the Supabase queue one at a time.
    Gracefully handles exceptions per company so that a single failure doesn't halt the pipeline.
    """
    print("==================================================")
    print("      SEC EDGAR Supply Chain Crawler Started      ")
    print("==================================================")
    
    success_count = 0
    failed_count = 0
    
    while True:
        # Fetch next company from the queue where status is 'not_started'
        res = supabase.table("company_queue")\
            .select("*")\
            .eq("status", "not_started")\
            .limit(1)\
            .execute()
            
        if not hasattr(res, "data") or not res.data:
            print("\n>>> No more companies in 'not_started' state. Crawler finished. <<<")
            break
            
        company_row = res.data[0]
        company_name = company_row["company_name"]
        
        print(f"\n[*] Processing Company: {company_name}")
        mark_status(supabase, company_name, "in_progress")
        
        try:
            # 1. Resolve Company Legal Name to SEC CIK
            cik = get_cik(company_name)
            if not cik:
                raise ValueError(f"Failed to find CIK for company: {company_name}")
            print(f"  [✓] Resolved CIK: {cik}")
            
            # 2. Get Recent Filings
            filings = get_filings(cik)
            if not filings:
                print(f"  [!] No recent filings (10-K, 20-F, 10-Q) found for CIK {cik}. Marking 'done'.")
                mark_status(supabase, company_name, "done")
                success_count += 1
                continue
            print(f"  [✓] Found {len(filings)} recent filing documents.")
            
            # 3. Process Each Filing Document
            company_total_sentences = 0
            for index, filing in enumerate(filings, start=1):
                accession = filing["accession"]
                form_type = filing["form"]
                filing_date = filing["date"]
                
                print(f"  --> [{index}/{len(filings)}] Fetching {form_type} filing ({filing_date}) [Accession: {accession}]...")
                
                try:
                    # Download primary document plain text
                    text = fetch_filing_text(cik, accession)
                    
                    # Segment and parse sentences
                    sentences = extract_sentences(text)
                    print(f"      [i] Found {len(sentences)} supply-chain-relevant sentences.")
                    
                    # Save rows to Supabase raw_sentences
                    saved = save_sentences(supabase, company_name, cik, accession, form_type, filing_date, sentences)
                    print(f"      [✓] Batch saved {saved} rows.")
                    company_total_sentences += saved
                    
                except Exception as doc_error:
                    # Log document failure but try to process remaining filings for the company if possible
                    print(f"      [✗] Error processing filing {accession}: {doc_error}")
            
            # Update status to done on successful resolution of company
            mark_status(supabase, company_name, "done")
            success_count += 1
            print(f"  [✓] Finished {company_name} | Total sentences saved: {company_total_sentences}")
            
        except Exception as e:
            # Graceful error handling per company to keep run uninterrupted
            print(f"  [✗] Error processing {company_name}: {e}")
            try:
                mark_status(supabase, company_name, "failed")
            except Exception as status_err:
                print(f"      Failed to update queue status to 'failed': {status_err}")
            failed_count += 1
            continue
            
    print("\n==================================================")
    print("                Crawler Run Summary               ")
    print("==================================================")
    print(f"  Companies successfully processed: {success_count}")
    print(f"  Companies failed:                 {failed_count}")
    print("==================================================")


if __name__ == "__main__":
    # Standard entry point execution when run directly
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[CRITICAL] SUPABASE_URL and SUPABASE_KEY environment variables are required.")
    else:
        # Initialize Supabase client
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        run_crawler(supabase_client)
