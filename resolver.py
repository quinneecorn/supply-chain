import os
import time
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from urllib.parse import quote_plus

# 1. Environment Setup
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OC_API_TOKEN = os.getenv("OPENCORPORATES_API_TOKEN") 

def get_opencorporates_data(company_name: str) -> dict | None:
    """
    Queries the OpenCorporates search API to find the canonical global identity of a company.
    """
    if not OC_API_TOKEN:
        time.sleep(1.5)
        
    search_query = quote_plus(company_name)
    url = f"https://api.opencorporates.com/v0.4/companies/search?q={search_query}"
    
    if OC_API_TOKEN:
        url += f"&api_token={OC_API_TOKEN}"

    try:
        response = requests.get(url, timeout=15)
        
        # Handle Rate Limiting gracefully
        if response.status_code == 403:
            print("    [!] OpenCorporates API rate limit hit (403).")
            return None
            
        response.raise_for_status()
        data = response.json()
        
        companies = data.get("results", {}).get("companies", [])
        
        if not companies:
            return None
            
        # For this architecture, we take the top match returned by their relevance engine
        best_match = companies[0].get("company", {})
        
        return {
            "legal_name": best_match.get("name"),
            "jurisdiction_code": best_match.get("jurisdiction_code"),
            "company_number": best_match.get("company_number"),
            "opencorporates_id": best_match.get("opencorporates_url")
        }
        
    except Exception as e:
        print(f"    [!] OpenCorporates API Error: {str(e)}")
        return None


def run_identity_resolver(supabase: Client):
    print("==================================================")
    print("      Global Identity Resolver Started            ")
    print("==================================================")
    
    while True:
        # Fetch companies that haven't been resolved yet (jurisdiction_code is NULL)
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
        
        oc_data = get_opencorporates_data(target_name)
        
        if oc_data:
            jurisdiction = oc_data["jurisdiction_code"]
            legal_name = oc_data["legal_name"]
            company_num = oc_data["company_number"]
            oc_id = oc_data["opencorporates_id"]
            
            # If it's a UK company, map the number to our uk_crn column
            uk_crn = company_num if jurisdiction == 'gb' else None
            
            print(f"  [✓] Found: {legal_name} ({jurisdiction})")
            if uk_crn:
                print(f"  [✓] UK CRN identified: {uk_crn}")
                
            # Update the database
            supabase.table("company_queue").update({
                "legal_name": legal_name,
                "jurisdiction_code": jurisdiction,
                "uk_crn": uk_crn,
                "opencorporates_id": oc_id
            }).eq("id", row_id).execute()
            
        else:
            print(f"  No exact match found on OpenCorporates.")
            supabase.table("company_queue").update({
                "jurisdiction_code": "unknown"
            }).eq("id", row_id).execute()

if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[CRITICAL] SUPABASE_URL and SUPABASE_KEY environment variables are required.")
    else:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        run_identity_resolver(supabase_client)