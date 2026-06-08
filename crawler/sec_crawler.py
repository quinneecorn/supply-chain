import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client
from .base_crawler import BaseCrawler

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "SupplyChainNLP/1.0 group7@hust.edu.vn")

class SECCrawler(BaseCrawler):
    def __init__(self, supabase: Client):
        super().__init__(supabase)
        self.headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate"
        }

    def execute_sec_request(self, url: str) -> requests.Response:
        time.sleep(0.11)
        retries = 3
        backoff_seconds = [5, 15, 60]
        
        for attempt in range(retries + 1):
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                if response.status_code == 429:
                    if attempt < retries:
                        time.sleep(backoff_seconds[attempt])
                        continue
                    else:
                        response.raise_for_status()
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt < retries:
                    time.sleep(backoff_seconds[attempt])
                else:
                    raise

    def get_cik(self, company_name: str) -> str | None:
        # no longer use a hardcoded overrides dictionary!
        # Wiki Resolver's canonical name.
        search_name = company_name 
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self.execute_sec_request(url)
            tickers_data = response.json()
            target_name_clean = search_name.lower().strip()
            
            for key, entry in tickers_data.items():
                if entry.get("title", "").lower().strip() == target_name_clean:
                    cik_val = entry.get("cik_str")
                    if cik_val: return str(cik_val).zfill(10)
                    
            for key, entry in tickers_data.items():
                if target_name_clean in entry.get("title", "").lower():
                    cik_val = entry.get("cik_str")
                    if cik_val: return str(cik_val).zfill(10)
        except Exception as e:
            print(f"  [CIK Lookup] Primary lookup error: {e}")
            
        try:
            url = f'https://efts.sec.gov/LATEST/search-index?q="{search_name}"&forms=10-K,20-F,S-1,S-1/A,424B1,424B2,424B3,424B4,424B5'
            response = self.execute_sec_request(url)
            hits = response.json().get("hits", {}).get("hits", [])
            if hits:
                return str(hits[0].get("_source", {}).get("cik")).zfill(10)
        except Exception:
            pass
        return None

    def get_filings(self, cik: str, form_types: tuple = ("10-K", "20-F", "S-1", "424"), max_filings: int = 10) -> list[dict]:
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        response = self.execute_sec_request(url)
        recent = response.json().get("filings", {}).get("recent", {})
        if not recent: return []
            
        accessions = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        
        filings = []
        for i, form in enumerate(forms):
            if any(form.startswith(t) for t in form_types):
                filings.append({
                    "accession": accessions[i].replace("-", ""),
                    "form": form,
                    "date": dates[i]
                })
                if len(filings) >= max_filings: break
        return filings

    def fetch_filing_text(self, cik: str, accession: str) -> str:
        accession_dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
        cik_stripped = cik.lstrip("0")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{accession}/{accession_dashed}-index.htm"
        
        index_soup = BeautifulSoup(self.execute_sec_request(index_url).text, "lxml")
        primary_doc_href = None
        
        for a in index_soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("/") and not href.startswith("/Archives/edgar/data/"): continue
            if href.endswith((".htm", ".html", ".txt")) and not href.endswith(("-index.htm", "-index.html")):
                primary_doc_href = href
                break
                
        if not primary_doc_href: return ""
            
        doc_url = primary_doc_href if primary_doc_href.startswith("http") else \
            (f"https://www.sec.gov{primary_doc_href}" if primary_doc_href.startswith("/") else \
             f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{accession}/{primary_doc_href}")
             
        doc_soup = BeautifulSoup(self.execute_sec_request(doc_url).text, "lxml")
        for element in doc_soup(["script", "style"]): element.extract()
        return doc_soup.get_text(separator=" ")

    def crawl(self, company_name: str, root_seed: str = None, sec_cik: str = None, company_id: int = None, **kwargs):
        print(f"\n[SECCrawler] Processing: {company_name}")
        try:
            cik = sec_cik or self.get_cik(company_name)
            if not cik:
                print(f"  Failed to find SEC CIK. Skipping.")
                return
            print(f"  Resolved CIK: {cik}")
            
            if company_id and not sec_cik:
                self.supabase.table("company_queue").update({"cik": cik}).eq("id", company_id).execute()
            
            filings = self.get_filings(cik)
            if not filings:
                print(f"  [!] No recent filings found.")
                return
                
            total_sentences = 0
            for i, filing in enumerate(filings, 1):
                print(f"  --> [{i}/{len(filings)}] Fetching {filing['form']} ({filing['date']})...")
                try:
                    text = self.fetch_filing_text(cik, filing["accession"])
                    sentences = self.extract_sentences(text)
                    if sentences:
                        acc = filing["accession"]
                        acc_dashed = f"{acc[:10]}-{acc[10:12]}-{acc[12:]}"
                        cik_stripped = cik.lstrip("0")
                        source_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc}/{acc_dashed}-index.htm"

                        # 2. Call the BaseCrawler's save_sentences with the root_seed!
                        saved = self.save_sentences(
                            company_name=company_name,
                            source_url=source_url,
                            doc_type=f"SEC {filing['form']}",
                            date=filing["date"],
                            sentences=sentences,
                            accession=acc,
                            root_seed=root_seed 
                        )
                        total_sentences += saved
                        print(f"      [✓] Batch saved {saved} supply-chain sentences.")
                except Exception as e:
                    print(f"      [✗] Error on {filing['accession']}: {e}")
            print(f"  [✓] Finished {company_name} | Total saved: {total_sentences}")
        except Exception as e:
            print(f"  [✗] SEC Crawler Error: {e}")