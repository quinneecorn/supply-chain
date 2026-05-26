from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv(".env")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

response = (
    supabase
    .table("company_queue")
    .select("*")
    .execute()
)

print(response.data)