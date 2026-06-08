import os
import requests
import tempfile
import fitz
from dotenv import load_dotenv
from supabase import create_client, Client
from .base_crawler import BaseCrawler

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CHS_API_KEY = os.getenv("CHS_API_KEY")

class CHSCrawler(BaseCrawler):
    def __init__(self, supabase: Client):
        super().__init__(supabase)

    def extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """Reads raw text from PDF bytes using PyMuPDF"""
        text_content = []
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf.flush()
            
            # Open and extract
            doc = fitz.open(temp_pdf.name)
            for page in doc:
                text_content.append(page.get_text())
            doc.close()
            
        return "\n".join(text_content)
    
    def get_crn(self, company_name: str) -> str | None:
        """Searches Companies House API by company name to find the CRN safely."""
        search_url = "https://api.company-information.service.gov.uk/search/companies"
        try:
            res = requests.get(search_url, params={"q": company_name}, auth=(CHS_API_KEY, ""), timeout=10)
            res.raise_for_status()
            items = res.json().get("items", [])
            
            if items:
                top_hit = items[0]
                crn = top_hit.get("company_number")
                title = top_hit.get("title", "Unknown Name")
                print(f"  [i] API Matched '{company_name}' to: {title} (CRN: {crn})")
                return crn
        except Exception as e:
            print(f"  [CRN Lookup] Failed to resolve name: {e}")
        return None

    def crawl(self, company_name: str, root_seed: str = None, uk_crn: str = None, company_id: int = None, **kwargs):
        print(f"\n[CHSCrawler] Initiating UK registry sweep for: {company_name}")
        
        if not CHS_API_KEY:
            print("  [!] Missing CHS_API_KEY. Skipping.")
            return
            
        crn = uk_crn or self.get_crn(company_name)

        if not crn:
            print("  [!] Could not find a valid UK CRN. Skipping.")
            return
            
        # Clean the CRN: Only zfill to 8 if it's purely numbers. Scottish (SC) or Northern Ireland (NI) CRNs will break if zfilled blindly.
        crn_str = str(crn).strip()
        clean_crn = crn_str.zfill(8) if crn_str.isdigit() else crn_str
        print(f"  [✓] Verified Target CRN: {clean_crn}")
        
        # Permanently save to DB
        if company_id and not uk_crn:
            try:
                self.supabase.table("company_queue").update({"uk_crn": clean_crn}).eq("id", company_id).execute()
            except Exception:
                pass

        url = f"https://api.company-information.service.gov.uk/company/{clean_crn}/filing-history"
        auth = (CHS_API_KEY, "") 

        try:
            # We pass the category as a param to ensure the URL perfectly resolves
            res = requests.get(url, params={"category": "accounts"}, auth=auth, timeout=10)
            res.raise_for_status()
            filings = res.json().get("items", [])
            
            if not filings:
                print(f"  [i] API returned 0 'accounts' filings for CRN {clean_crn}.")
                return

            print(f"  [✓] Found {len(filings)} accounts filings. Ripping the PDFs...")
            total_company_sentences = 0
            
            for index, filing in enumerate(filings[:5], start=1):
                filing_date = filing.get("date", "1970-01-01")
                doc_metadata_url = filing.get("links", {}).get("document_metadata")
                
                if not doc_metadata_url:
                    continue
                    
                print(f"    -> [{index}] Downloading PDF for filing dated {filing_date}...")
                
                doc_url = f"{doc_metadata_url}/content"
                doc_res = requests.get(doc_url, auth=auth, headers={"Accept": "application/pdf"}, allow_redirects=True, timeout=15)
                
                if doc_res.status_code == 200 and "%PDF" in doc_res.content[:10].decode(errors="ignore"):
                    pdf_text = self.extract_pdf_text(doc_res.content)
                    sentences = self.extract_sentences(pdf_text)
                    
                    if sentences:
                        saved = self.save_sentences(
                            company_name=company_name,
                            source_url=doc_url,
                            doc_type="UK Strategic Report",
                            date=filing_date,
                            sentences=sentences,
                            root_seed=root_seed 
                        )
                        total_company_sentences += saved
                        print(f"       [+] Scraped and saved {saved} supply chain sentences.")
                else:
                    print(f"       [✗] PDF Download Failed (Status {doc_res.status_code} or bad file signature).")
                    
            print(f"  [✓] Finished {company_name} | Total sentences saved: {total_company_sentences}")
            
        except Exception as e:
            print(f"  [✗] CHS Crawler Error: {str(e)}")


if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[CRITICAL] Database environment variables missing.")
    else:
        db_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        crawler = CHSCrawler(db_client)