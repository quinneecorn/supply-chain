import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

print("Peeking at the table structure...\n")

response = supabase.table("scraped_sentences").select("*").limit(1).execute()

if response.data:
    columns = list(response.data[0].keys())
    print("🎯 You need to provide data for these columns:")
    for col in columns:
        print(f" - {col}")
else:
    print("Table is empty, cannot infer columns from data!")