# database/peek_tables.py

from supabase_client import supabase

print("Peeking at the table structure...\n")

response = (
    supabase
    .table("scraped_sentences")
    .select("*")
    .limit(1)
    .execute()
)

if response.data:

    columns = list(response.data[0].keys())

    print("🎯 You need to provide data for these columns:")

    for col in columns:
        print(f" - {col}")

else:
    print("Table is empty, cannot infer columns from data!")