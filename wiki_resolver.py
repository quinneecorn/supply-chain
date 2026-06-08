import os
import time
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from urllib.parse import quote

# 1. Environment Setup
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Wikimedia strictly requires a descriptive User-Agent
USER_AGENT = "SupplyChainCrawler/1.0 (group7@hust.edu.vn)"

def get_wiki_data(company_name: str) -> dict | None:
    """
    Queries the public Wikipedia REST API to find company metadata.
    """
    time.sleep(1.0) # Polite rate limit
    
    headers = {"User-Agent": USER_AGENT}
    search_query = quote(company_name)
    
    # Step 1: Search for the exact Wikipedia page title
    search_url = f"https://en.wikipedia.org/w/rest.php/v1/search/title?q={search_query}&limit=1"
    
    try:
        search_res = requests.get(search_url, headers=headers, timeout=10)
        search_res.raise_for_status()
        pages = search_res.json().get("pages", [])
        
        if not pages:
            return None
            
        page_title = pages[0].get("key")
        
        # Step 2: Grab the summary snippet to infer jurisdiction/country
        summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={page_title}&format=json"
        summary_res = requests.get(summary_url, headers=headers, timeout=10)
        
        pages_data = summary_res.json().get("query", {}).get("pages", {})
        extract_text = ""
        for page_id, info in pages_data.items():
            if page_id != "-1":
                extract_text = info.get("extract", "")
                break
                
        # Simple heuristic to guess jurisdiction based on common Wiki intros
        # You can expand this dictionary as needed
        jurisdiction = "unknown"
        extract_lower = extract_text.lower()
        
        if "american" in extract_lower or "united states" in extract_lower:
            jurisdiction = "us"
        elif "british" in extract_lower or "united kingdom" in extract_lower:
            jurisdiction = "gb"
        elif "taiwanese" in extract_lower or "taiwan" in extract_lower:
            jurisdiction = "tw"
        elif "japanese" in extract_lower or "japan" in extract_lower:
            jurisdiction = "jp"
        elif "dutch" in extract_lower or "netherlands" in extract_lower:
            jurisdiction = "nl"
        elif "korean" in extract_lower or "south korea" in extract_lower:
             jurisdiction = "kr"
             
        return {
            "legal_name": pages[0].get("title"), # The official Wiki page title
            "jurisdiction_code": jurisdiction,
            "opencorporates_id": f"https://en.wikipedia.org/wiki/{page_title}" # Store wiki URL here for now
        }
        
    except Exception as e:
        print(f"    [!] Wiki API Error: {str(e)}")
        return None

def run_identity_resolver(supabase: Client):
    print("==================================================")
    print("      Global Identity Resolver (Wiki Engine)      ")
    print("==================================================")
    
    # 1. We must explicitly reset the 'unknown' statuses the failed script just made
    print("Resetting previously failed attempts...")
    supabase.table("company_queue").update({"jurisdiction_code": None}).eq("jurisdiction_code", "unknown").execute()
    
    while True:
        res = supabase.table("company_queue")\
            .select("id, company_name")\
            .is_("jurisdiction_code", "null")\
            .limit(1)\
            .execute()
            
        if not hasattr(res, "data") or not res.data:
            print("\n>>> All companies in queue have been globally resolved. <<<")
            break
            
        row = res.data[0]
        row_id = row["id"]
        target_name = row["company_name"]
        
        print(f"\n[*] Resolving: {target_name}")
        
        wiki_data = get_wiki_data(target_name)
        
        if wiki_data:
            jurisdiction = wiki_data["jurisdiction_code"]
            legal_name = wiki_data["legal_name"]
            wiki_url = wiki_data["opencorporates_id"]
            
            print(f"  [✓] Found: {legal_name} (Jurisdiction: {jurisdiction})")
                
            supabase.table("company_queue").update({
                "legal_name": legal_name,
                "jurisdiction_code": jurisdiction,
                "opencorporates_id": wiki_url
            }).eq("id", row_id).execute()
            
        else:
            print(f"  [✗] No exact match found on Wikipedia.")
            supabase.table("company_queue").update({
                "jurisdiction_code": "unknown" # mark unknown so we skip it next time
            }).eq("id", row_id).execute()

if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[CRITICAL] SUPABASE_URL and SUPABASE_KEY environment variables are required.")
    else:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        run_identity_resolver(supabase_client)