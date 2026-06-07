import urllib.request
import json
import os

# 1. Put your source project URL and Key here
url = os.getenv("SUPABASE_URL", "https://iwmfgmeoxoptasrsrlom.supabase.co")
if not url.endswith("/rest/v1/"):
    url += "/rest/v1/"
anon_key = os.getenv("SUPABASE_KEY", "")

def discover_database():
    print("Connecting to Supabase API to discover schema...")
    
    # Set up the request with your API credentials
    req = urllib.request.Request(url)
    req.add_header("apikey", anon_key)
    req.add_header("Authorization", f"Bearer {anon_key}")
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
            # The API exposes all accessible tables under 'definitions'
            definitions = data.get('definitions', {})
            
            if not definitions:
                print("Connected successfully, but no tables are exposed to the public API.")
                return
                
            print("\n✅ DISCOVERED TABLES AND COLUMNS:\n")
            
            # Loop through and print every table and its columns
            for table_name, schema in definitions.items():
                print(f"📦 Table: {table_name}")
                properties = schema.get('properties', {})
                
                for col_name, col_details in properties.items():
                    # Extract the data type (e.g., text, uuid, bigint)
                    col_type = col_details.get('format') or col_details.get('type', 'unknown')
                    print(f"   - {col_name} ({col_type})")
                    
                print("-" * 40)
                
    except urllib.error.URLError as e:
        print(f"Error connecting to the API: {e}")

if __name__ == "__main__":
    discover_database()