# seed_companies.py — DO NOT MODIFY
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
if not URL or not KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in your environment or .env file.")

supabase: Client = create_client(URL, KEY)

SEED_COMPANIES = [
    "Apple Inc.", "Microsoft Corporation", "Sony Group Corporation",
    "Dell Technologies Inc.", "HP Inc.", "Samsung Electronics Co., Ltd.",
    "Taiwan Semiconductor Manufacturing Company Limited",
    "Intel Corporation", "Advanced Micro Devices, Inc.", "NVIDIA Corporation",
    "Broadcom Inc.", "Qualcomm Incorporated", "Texas Instruments Incorporated",
    "Micron Technology, Inc.", "ASML Holding N.V."
]

def plant_seeds():
    print(f"Planting seed companies to Supabase URL: {URL}")
    for company in SEED_COMPANIES:
        try:
            supabase.table("company_queue").insert({
                "company_name": company,
                "naics_code": "334",
                "tier_level": 0,
                "status": "not_started"
            }).execute()
            print(f"Successfully planted: {company}")
        except Exception as e:
            print(f"Failed to plant {company}. Error: {e}")

if __name__ == "__main__":
    plant_seeds()
