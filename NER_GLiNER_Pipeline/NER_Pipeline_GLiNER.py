import os
import json
from supabase import create_client, Client

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "SUPABASE_URL": os.getenv("SUPABASE_URL", "https://hihfkqvheqbjckmtlkvh.supabase.co"),
    "SUPABASE_KEY": os.getenv("SUPABASE_KEY", ""),
    "TABLE_NAME": os.getenv("TABLE_NAME", "NLP Gliner"),
    "MODEL_NAME": "urchade/gliner_medium-v2.1",
    "BATCH_SIZE": 1000 # Max recommended for single API call
}

# 1. Initialize Supabase
supabase: Client = create_client(CONFIG["SUPABASE_URL"], CONFIG["SUPABASE_KEY"])

# 2. Initialize GLiNER (Lazy loading)
_model = None

def get_model():
    global _model
    if _model is None:
        from gliner import GLiNER
        print(f"Loading GLiNER model ({CONFIG['MODEL_NAME']})...")
        _model = GLiNER.from_pretrained(CONFIG["MODEL_NAME"])
    return _model

def process_with_gliner(text):
    """Identifies entities locally using GLiNER and returns structured data."""
    model = get_model()
    
    # Refined labels that work better with gliner_medium
    labels = [
        "Company Name", "Public Company", "Private Company", 
        "Government Body", "Brand Name", "Product Model", "Project Name"
    ]
    
    # Groups for mapping to FROM/TO tokens
    FROM_LABELS = ["Company Name", "Public Company", "Private Company", "Government Body"]
    TO_LABELS = ["Brand Name", "Product Model", "Project Name"]

    # Special Role Mappings
    ROLE_MAPPING = {
        "Acquiring Corporation": "[__NE_TO__]",
        "Recipient": "[__NE_TO__]",
        "Department": "[__NE_FROM__]",
        "The Committee": "[__NE_FROM__]"
    }

    try:
        # LOWERED THRESHOLD TO 0.3
        entities = model.predict_entities(text, labels, threshold=0.3)
    except Exception:
        return None
    
    # Filter for Proper Names and handle overlaps
    proper_entities = []
    # Sort by score so we keep the highest confidence if spans overlap
    entities = sorted(entities, key=lambda x: x['score'], reverse=True)
    
    seen_indices = set()
    for ent in entities:
        name = ent['text'].strip()
        start, end = ent['start'], ent['end']
        
        # Check if span overlaps with a higher-scoring entity
        if any(i in seen_indices for i in range(start, end)):
            continue
            
        # Basic proper name check: Starts with Uppercase
        if name and name[0].isupper():
            proper_entities.append(ent)
            for i in range(start, end):
                seen_indices.add(i)

    # Sort entities by start index (descending) to avoid index shift during masking
    sorted_entities = sorted(proper_entities, key=lambda x: x['start'], reverse=True)
    
    masked_sentence = text
    from_entities = []
    to_entities = []
    primary_from = None
    primary_to = None

    for ent in sorted_entities:
        name = ent['text']
        label = ent['label']
        start, end = ent['start'], ent['end']
        
        token = ""
        # Check special role mapping first
        if name in ROLE_MAPPING:
            token = ROLE_MAPPING[name]
        elif label in FROM_LABELS:
            token = "[__NE_FROM__]"
        elif label in TO_LABELS:
            token = "[__NE_TO__]"
        
        if token:
            if token == "[__NE_FROM__]":
                from_entities.append(name)
                primary_from = name
            else:
                to_entities.append(name)
                primary_to = name
            masked_sentence = masked_sentence[:start] + token + masked_sentence[end:]

    return {
        "masked_sentence": masked_sentence,
        "entity_from": primary_from,
        "entity_to": primary_to,
        "extracted_entities": {
            "from_entities": list(set(from_entities)),
            "to_entities": list(set(to_entities))
        }
    }

from concurrent.futures import ThreadPoolExecutor

def update_database(record_id, result):
    """Worker function to push a single update to Supabase."""
    try:
        supabase.table(CONFIG["TABLE_NAME"]) \
            .update({
                "masked_sentence": result["masked_sentence"],
                "entity_from": result["entity_from"],
                "entity_to": result["entity_to"],
                "extracted_entities": result["extracted_entities"],
                "masked_status": "completed"
            }) \
            .eq("id", record_id) \
            .execute()
        return True
    except Exception as e:
        print(f"Update error for ID {record_id}: {e}")
        return False

def run_pipeline():
    print(f"Starting Multi-Stream pipeline on table: {CONFIG['TABLE_NAME']}")
    
    # We use a ThreadPool for database updates to avoid waiting on network I/O
    # 10 workers is a safe balance for Supabase connection limits
    with ThreadPoolExecutor(max_workers=10) as executor:
        while True:
            print(f"\nFetching next batch of {CONFIG['BATCH_SIZE']} pending records...")
            try:
                response = supabase.table(CONFIG["TABLE_NAME"]) \
                    .select("*") \
                    .eq("masked_status", "not_started") \
                    .order("id") \
                    .limit(CONFIG["BATCH_SIZE"]) \
                    .execute()

                records = response.data

                if not records:
                    print("✅ All records processed! No more pending rows found.")
                    break

                print(f"Processing {len(records)} records locally...")
                
                # GLiNER model processing stays sequential (it's CPU/GPU intensive)
                # but we submit the database update to the background thread pool
                for record in records:
                    result = process_with_gliner(record["raw_sentence"])
                    
                    if result:
                        # Submit update task to the thread pool (Non-blocking)
                        executor.submit(update_database, record["id"], result)
                
                print(f"Batch inference complete. Updates are streaming to database in background...")

            except Exception as e:
                print(f"Error in batch: {e}")
                break

if __name__ == "__main__":
    run_pipeline()
