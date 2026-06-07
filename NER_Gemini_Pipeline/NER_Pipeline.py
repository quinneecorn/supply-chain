import os
from supabase import create_client, Client
from openai import OpenAI
import json

from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURATION - Updated for New Database
# ==========================================
CONFIG = {
    "SUPABASE_URL": os.getenv("SUPABASE_URL", "https://nrgvharhfbifnpfeagte.supabase.co"),
    "SUPABASE_KEY": os.getenv("SUPABASE_KEY", ""),
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
    "MODEL_NAME": "google/gemini-2.0-flash-001",
    "BATCH_SIZE": 100,  # Smaller batches for smoother parallel streaming
    "MAX_WORKERS": 10   # Number of concurrent AI streams
}

# 1. Initialize Clients
supabase: Client = create_client(CONFIG["SUPABASE_URL"], CONFIG["SUPABASE_KEY"])

# OpenRouter uses the OpenAI-compatible API format
ai_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=CONFIG["OPENROUTER_API_KEY"],
)

def get_label_from_ai(text):
    """Sends raw text to OpenRouter and returns structured masking results for corporate/legal data."""
    system_prompt = ( """You are an expert Named Entity Recognition (NER) system specializing in corporate, legal, and contractual documents.
    Your task is to identify entities acting as the 'Source' (FROM) and the 'Recipient/Product' (TO) and return a JSON object.

    DEFINITIONS:
    - [__NE_FROM__] (Source/Parent): The entity providing rights, shares, or employment (e.g., 'The Company', 'The Committee'); the seller/acquired company in a merger; the parent company; the manufacturer or originator of a technology.
    - [__NE_TO__] (Recipient/Destination/Product): The entity receiving rights or making a purchase (e.g., 'Acquiring Corporation', 'The Recipient'); the buyer; the specific product, service, joint venture, or facility being created or discussed.

    STRICT RULES:
    1. Acquisitions: The Buyer/Acquirer is [__NE_TO__]. The Target/Seller is [__NE_FROM__].
    2. Contracts/Awards: The Granting body (Committee/Company) is [__NE_FROM__]. The receiving entity/product is [__NE_TO__].
    3. Business Risks: The company facing the risk (e.g., Sony) is [__NE_FROM__]. The market, industry, or competing technology it interacts with is [__NE_TO__].
    4. Entity Extraction: Extract the exact names of the primary entities into 'entity_from' and 'entity_to'. List all secondary entities in the 'extracted_entities' arrays.
    5. Sentence Integrity: Do NOT alter the sentence structure or punctuation.
    6. Output: Return ONLY the raw JSON object.

    EXAMPLES:
    Input: In the event of a Change in Control, the Acquiring Corporation may assume the Company’s rights.
    Output: {
        "masked_sentence": "In the event of a Change in Control, the [__NE_TO__] may assume the [__NE_FROM__]’s rights.",
        "entity_from": "The Company",
        "entity_to": "Acquiring Corporation",
        "extracted_entities": {"from_entities": ["The Company"], "to_entities": ["Acquiring Corporation"]}
    }

    Input: Sony completed the acquisition of Altair Semiconductor, which develops LTE technologies.
    Output: {
        "masked_sentence": "[__NE_TO__] completed the acquisition of [__NE_FROM__], which develops [__NE_TO__] technologies.",
        "entity_from": "Altair Semiconductor",
        "entity_to": "Sony",
        "extracted_entities": {"from_entities": ["Altair Semiconductor"], "to_entities": ["Sony", "LTE"]}
    }
    """
    )
    try:
        response = ai_client.chat.completions.create(
            model=CONFIG["MODEL_NAME"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Process this: {text}"}
            ],
            temperature=0,
            response_format={ "type": "json_object" },
            max_tokens=1000
        )
        data = json.loads(response.choices[0].message.content)
        
        # If AI returns a list (complex sentence), we return a special signal to mark it in DB
        if isinstance(data, list):
            return "SKIPPED_MULTI"
        
        if isinstance(data, dict):
            return data
        
        return None
    except Exception as e:
        print(f"AI Error: {e}")
        return None

def process_record(record):
    """Worker function for parallel processing."""
    record_id = record['id']
    print(f"Processing ID {record_id}...")
    
    # 3. Get AI structured data
    result = get_label_from_ai(record["raw_sentence"])
    
    if result == "SKIPPED_MULTI":
        # Store the skip so we don't process it again
        supabase.table("NLP_copy_test") \
            .update({"masked_status": "skipped_multi"}) \
            .eq("id", record_id) \
            .execute()
        print(f"ID {record_id} marked as 'skipped_multi'.")
        return

    if result:
        # 4. Push update back to specific columns
        try:
            supabase.table("NLP_copy_test") \
                .update({
                    "masked_sentence": result.get("masked_sentence"),
                    "entity_from": result.get("entity_from"),
                    "entity_to": result.get("entity_to"),
                    "extracted_entities": result.get("extracted_entities"),
                    "masked_status": "completed"
                }) \
                .eq("id", record_id) \
                .execute()
            print(f"✅ ID {record_id} updated.")
        except Exception as e:
            print(f"Database update error for ID {record_id}: {e}")

def run_pipeline():
    print(f"Starting Multi-Stream LLM pipeline on table: NLP_copy_test")
    
    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        while True:
            print(f"\nFetching next batch of {CONFIG['BATCH_SIZE']} pending records...")
            try:
                response = supabase.table("NLP_copy_test") \
                    .select("*") \
                    .eq("masked_status", "not_started") \
                    .order("id") \
                    .limit(CONFIG["BATCH_SIZE"]) \
                    .execute()

                records = response.data

                if not records:
                    print("✅ All records processed! No more pending rows found.")
                    break

                # Map the worker function across the records batch
                executor.map(process_record, records)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                break

if __name__ == "__main__":
    run_pipeline()
