import os
import sys
import time
from dotenv import load_dotenv
from supabase import create_client, Client

from wiki_resolver import get_wiki_data
from crawler.sec_crawler import SECCrawler
from crawler.chs_crawler import CHSCrawler
from crawler.newsapi_crawler import NewsCrawler

# Adjust this import based on the actual function name in your pipeline.py
from gemma.pipeline import run_pipeline_batch 

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[CRITICAL] Missing Supabase credentials in .env")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def inject_seed_company(company_name: str) -> dict:
    """Checks if a company exists. If not, resolves its identity and injects it as Tier 0."""
    print(f"\n[*] Checking database for '{company_name}'...")
    
    # 1. Quick check with raw input
    res = supabase.table("company_queue").select("*").ilike("company_name", company_name).execute()
    if hasattr(res, "data") and res.data:
        print(f"  [i] '{res.data[0]['company_name']}' is already in the database (Tier {res.data[0].get('tier_level')}).")
        return res.data[0]
        
    print(f"  [i] New company detected. Resolving global identity via Wikipedia...")
    
    wiki_data = get_wiki_data(company_name)
    if wiki_data and wiki_data.get("legal_name"):
        canonical_name = wiki_data["legal_name"]
        jurisdiction = wiki_data.get("jurisdiction_code")
        wiki_url = wiki_data.get("opencorporates_id")
        print(f"  [✓] Identity resolved: {canonical_name} (Jurisdiction: {jurisdiction})")
    else:
        print(f"  [!] Could not resolve canonical identity. Using raw input.")
        canonical_name = company_name
        jurisdiction = None
        wiki_url = None

    # 3. Check again with canonical name
    res_canon = supabase.table("company_queue").select("*").ilike("company_name", canonical_name).execute()
    if not res_canon.data:
        res_canon = supabase.table("company_queue").select("*").ilike("legal_name", canonical_name).execute()

    if hasattr(res_canon, "data") and res_canon.data:
         print(f"  [i] Canonical identity '{canonical_name}' already exists in database (Tier {res_canon.data[0].get('tier_level')}).")
         return res_canon.data[0]

    # 4. Insert as Tier 0 Seed
    assign_status = "not_started" if jurisdiction else "gray_zone"
    
    payload = {
        "company_name": canonical_name,
        "legal_name": canonical_name,
        "tier_level": 0,
        "status": assign_status,
        "root_seed": canonical_name,
        "jurisdiction_code": jurisdiction,
        "opencorporates_id": wiki_url
    }
    
    insert_res = supabase.table("company_queue").insert(payload).execute()
    print(f"  [+] Successfully injected {canonical_name} as a Tier 0 Seed Company!")
    return insert_res.data[0]


def run_crawlers_for_company(company_data: dict):
    """Forces the crawlers to run immediately for the specified company."""
    company_name = company_data["company_name"]
    jurisdiction = company_data.get("jurisdiction_code")
    uk_crn = company_data.get("uk_crn")
    
    root_seed = company_data.get("root_seed")
    company_id = company_data["id"]
    
    print(f"\n  >>> Scraping: {company_name}")
    supabase.table("company_queue").update({"status": "in_progress"}).eq("id", company_data["id"]).execute()
    
    NewsCrawler(supabase).crawl(company_name, root_seed=root_seed)
    
    if jurisdiction == "gb":
        CHSCrawler(supabase).crawl(company_name, uk_crn=uk_crn, root_seed=root_seed, company_id=company_id)
    else:
        SECCrawler(supabase).crawl(company_name, root_seed=root_seed, company_id=company_id)
        
    supabase.table("company_queue").update({"status": "completed"}).eq("id", company_data["id"]).execute()


def run_bfs_trace(max_scrape_tier: int = 2):
    """
    Executes a Breadth-First Search across the database.
    If max_scrape_tier is 2:
    - Scrapes Tier 0 -> LLM Discovers Tier 1
    - Scrapes Tier 1 -> LLM Discovers Tier 2
    - Scrapes Tier 2 -> LLM Discovers Tier 3
    """
    print(f"   INITIATING BREADTH-FIRST SEARCH (UP TO TIER {max_scrape_tier})")
    
    for current_tier in range(max_scrape_tier + 1):
        print(f"   BFS LEVEL: PROCESSING TIER {current_tier}")
    
        while True:
            res = supabase.table("company_queue").select("*").eq("status", "not_started").eq("tier_level", current_tier).execute()
            pending_companies = res.data if hasattr(res, "data") else []
            
            if not pending_companies:
                print(f"[i] No more pending companies at Tier {current_tier} to scrape.")
                break
                
            print(f"[i] Found {len(pending_companies)} companies to scrape at Tier {current_tier}.")
            for company in pending_companies:
                run_crawlers_for_company(company)
        
        print(f"\n[i] Scraping complete for Tier {current_tier}. Waking up Pipeline...")
        print(f"[i] Hunting for Tier {current_tier + 1} suppliers...")
        time.sleep(1)
        
        try:
            # LLM will read the sentences and push newly discovered companies 
            # to the queue at `tier_level = current_tier + 1`
            run_pipeline_batch() 
            print(f"[✓] Extraction for Tier {current_tier} Complete!")
        except Exception as e:
            print(f"\n[!] Pipeline encountered an error: {e}")
            
    print(f"   BFS TRACE COMPLETE")
    print(f"Target reached: Your database now contains up to Tier {max_scrape_tier + 1} suppliers.")
    
def review_gray_zone():
    """Fetches all quarantined companies and lets you approve or delete them."""
    print("\n--- ENTERING THE GRAY ZONE ---")
    res = supabase.table("company_queue").select("*").eq("status", "gray_zone").execute()
    
    gray_companies = res.data if hasattr(res, "data") else []
    
    if not gray_companies:
        print("[i] The Gray Zone is empty.")
        return
        
    print(f"[!] Found {len(gray_companies)} companies missing a jurisdiction.\n")
    
    for comp in gray_companies:
        comp_id = comp['id']
        name = comp['company_name']
        tier = comp['tier_level']
        
        print(f"Company: {name} (Tier {tier})")
        action = input("  [A]pprove (Add to Table) | [D]elete | [S]kip : ").strip().lower()
        
        if action == 'a':
            supabase.table("company_queue").update({"status": "not_started"}).eq("id", comp_id).execute()
            print(f"  [+] {name} moved to 'not_started' queue.")
        elif action == 'd':
            supabase.table("company_queue").delete().eq("id", comp_id).execute()
            print(f"  [-] {name} deleted from database.")
        elif action == 'q':
            print("Exiting Gray Zone...")
            break
        else:
            print(f"  [~] Skipped {name}. It remains in the Gray Zone.")


def main_loop():
    print(r"""
        Supply Chain Tracer
    """)
    
    while True:
        target = input("\nEnter a Seed Company to begin BFS tracing (or 'quit' to exit): ").strip()
        
        if target.lower() in ['q', 'quit', 'exit']:
            break
        elif target.lower() == 'review':
            review_gray_zone() 
            continue
        if not target:
            continue
            
        # 1. Inject the Seed
        inject_seed_company(target)
    
        run_bfs_trace(max_scrape_tier=2)

if __name__ == "__main__":
    main_loop()
    

