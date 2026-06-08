import os
import requests
from dotenv import load_dotenv

load_dotenv()
URL = os.getenv("SUPABASE_URL").rstrip('/') 
KEY = os.getenv("SUPABASE_KEY")

print("Downloading the database blueprint...\n")

blueprint_url = f"{URL}/rest/v1/"

# The magic headers
headers = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Accept": "application/openapi+json" 
}

response = requests.get(blueprint_url, headers=headers)

if response.status_code == 200:
    blueprint = response.json()
    target_table = "scraped_sentences"
    
    if target_table in blueprint.get('definitions', {}):
        columns = blueprint['definitions'][target_table]['properties'].keys()
        print(f"🎯 Columns for empty table '{target_table}':")
        for col in columns:
            print(f" - {col}")
    else:
        print(f"Table '{target_table}' not found in blueprint. Did you spell it exactly right in Supabase?")
else:
    print(f"Failed to get blueprint. Status: {response.status_code}")
    print(f"Reason: {response.text}")