import os
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. Securely load the database keys from your .env file
load_dotenv()
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(URL, KEY)

# 2. The Anchor Seeds (Top Consumer Electronics & Semiconductor Companies)
SEED_COMPANIES = [
    "Apple Inc.",
    "Microsoft Corporation",
    "Sony Group Corporation",
    "Dell Technologies Inc.",
    "HP Inc.",
    "Samsung Electronics Co., Ltd.",
    "Taiwan Semiconductor Manufacturing Company Limited", # TSMC
    "Intel Corporation",
    "Advanced Micro Devices, Inc.", # AMD
    "NVIDIA Corporation",
    "Broadcom Inc.",
    "Qualcomm Incorporated",
    "Texas Instruments Incorporated",
    "Micron Technology, Inc.",
    "ASML Holding N.V."
]

def plant_seeds():
    print(f"🚀 Planting {len(SEED_COMPANIES)} seed nodes into Supabase...")
    
    for company in SEED_COMPANIES:
        try:
            # Insert the company into the queue as a Tier 1 seed
            supabase.table("company_queue").insert({
                "company_name": company,
                "naics_code": "334", # Computer and Electronic Product Manufacturing
                "tier_level": 0,
                "status": "not_started"
            }).execute()
            print(f"🌱 Planted: {company}")
            
        except Exception as e:
            # Our SQL table has a UNIQUE constraint on company_name.
            # If you run this script twice, it will safely skip duplicates instead of crashing.
            print(f"Failed to plant {company}. Error: {e}")
            
    print("\n✅ Seeding complete! The queue is primed and ready for the crawlers.")

if __name__ == "__main__":
    plant_seeds()