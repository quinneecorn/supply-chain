import os
import time
from dotenv import load_dotenv
from supabase import create_client, Client

from crawler.sec_crawler import SECCrawler
from crawler.chs_crawler import CHSCrawler
from crawler.newsapi_crawler import NewsCrawler 

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def mark_status(supabase: Client, company_name: str, status: str) -> None:
    """Updates the status in the company_queue table."""
    supabase.table("company_queue").update({"status": status}).eq("company_name", company_name).execute()

def run_dispatcher(supabase: Client):
    print("==================================================")
    print("      Global Crawler Dispatcher Started           ")
    print("==================================================")
    
    # Initialize our crawler tools
    sec_crawler = SECCrawler(supabase)
    chs_crawler = CHSCrawler(supabase)
    news_crawler = NewsCrawler(supabase)
    
    while True:
        res = supabase.table("company_queue")\
            .select("*")\
            .eq("status", "not_started")\
            .limit(1)\
            .execute()
            
        if not hasattr(res, "data") or not res.data:
            print("\n>>> No more companies in 'not_started' state. Dispatcher finished. <<<")
            break
            
        row = res.data[0]
        company_name = row["company_name"]
        jurisdiction = row.get("jurisdiction_code")
        uk_crn = row.get("uk_crn")
        
        print(f"\n[*] Dispatching Target: {company_name}")
        mark_status(supabase, company_name, "in_progress")
        
        try:
            news_crawler.crawl(company_name)
            if jurisdiction == "gb" and uk_crn:
                chs_crawler.crawl(company_name, uk_crn=uk_crn)
                
            elif jurisdiction == "us":
                sec_crawler.crawl(company_name)
                
            else:
                print(f"  [i] Jurisdiction is {jurisdiction}. Attempting SEC EDGAR as fallback.")
                sec_crawler.crawl(company_name)
            
            mark_status(supabase, company_name, "completed")
            print(f"[✓] {company_name} fully processed by all relevant crawlers.")
            
        except Exception as e:
            print(f"[✗] Global routing error for {company_name}: {e}")
            mark_status(supabase, company_name, "failed")
            
        time.sleep(1)

if __name__ == "__main__":
    db_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    run_dispatcher(db_client)