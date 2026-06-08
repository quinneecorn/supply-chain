# reset_queue.py
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in your environment or .env file.")

supabase: Client = create_client(URL, KEY)

def reset_queue():
    print(f"Connecting to Supabase: {URL}")
    print("Resetting all company statuses in company_queue back to 'not_started'...")
    try:
        # Update all rows
        response = supabase.table("company_queue").update({
            "status": "not_started"
        }).neq("status", "not_started").execute()
        
        if hasattr(response, "data"):
            print(f"Successfully reset {len(response.data)} companies back to 'not_started'.")
        else:
            print("Successfully updated company queue.")
    except Exception as e:
        print(f"Error resetting company queue: {e}")

if __name__ == "__main__":
    reset_queue()
