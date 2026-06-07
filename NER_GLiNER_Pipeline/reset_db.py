import os
from supabase import create_client, Client

# ==========================================
# CONFIGURATION - GLiNER NER Pipeline
# ==========================================
CONFIG = {
    "SUPABASE_URL": os.getenv("SUPABASE_URL", "https://hihfkqvheqbjckmtlkvh.supabase.co"),
    "SUPABASE_KEY": os.getenv("SUPABASE_KEY", ""), 
}

supabase: Client = create_client(CONFIG["SUPABASE_URL"], CONFIG["SUPABASE_KEY"])

def reset_database():
    print("Connecting to Supabase to reset GLiNER progress...")
    
    try:
        # Reset specific columns and set status back to 'not_started'
        # Targets 'NLP Gliner' table
        response = supabase.table("NLP Gliner") \
            .update({
                "masked_sentence": None,
                "extracted_entities": None,
                "entity_from": None,
                "entity_to": None,
                "masked_status": "not_started"
            }) \
            .neq("masked_status", "not_started") \
            .execute()
        
        count = len(response.data) if response.data else 0
        print(f"✅ Success! {count} records have been reset to 'not_started'.")
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")

if __name__ == "__main__":
    reset_database()
