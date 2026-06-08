import os
import sys
import json
import time
from google import genai
from google.genai import types
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
OLD_SUPABASE_URL = os.getenv("SUPABASE_URL_0")
OLD_SUPABASE_KEY = os.getenv("SUPABASE_KEY_0")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"

BATCH_SIZE = 50 
MAX_LIMIT = 4000  # Hard stop after exactly 8000 rows

missing_vars = []
if not OLD_SUPABASE_URL: missing_vars.append("OLD_SUPABASE_URL")
if not OLD_SUPABASE_KEY: missing_vars.append("OLD_SUPABASE_KEY")
if not GEMINI_API_KEY: missing_vars.append("GEMINI_API_KEY")

if missing_vars:
    print(f"[!] Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Initialize Clients
supabase: Client = create_client(OLD_SUPABASE_URL, OLD_SUPABASE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are an expert supply chain data extractor. Analyze the following batch of sentences and extract supply chain relationships.
You must return a JSON object containing a "results" array. 

ABSOLUTE ZERO-TOLERANCE RULES FOR ENTITIES (NO GHOST ENTITIES):
1. PROPER NOUNS ONLY: `entity_from` and `entity_to` MUST be specific, real-world named companies (e.g., "Apple", "Foxconn", "TSMC").
2. FORBIDDEN WORDS: You are strictly forbidden from using generic nouns, pronouns, or roles (e.g., "banks", "the supplier", "customers", "we", "the company", "our partners", "third-party").
3. THE NULL RULE: If a specific company name is missing for either the buyer or the supplier, you MUST set that field to null.
4. NO NAMES = NO RELATION: If either entity is null or a generic word, you MUST set relation_id to 0 and has_explicit_names to false.

RELATIONSHIP ID MAPPING (CRITICAL):
You are given the 'source_company' (Company A) and you must find the other specific company (Company B). You MUST use one of the following integers for `relation_id`:
- 0: No clear relationship, or one entity is missing/generic.
- 1: Company B supplies Company A (B -> A)
- 2: Company A supplies Company B (A -> B)
- 3: Partnership / Collaboration
- 4: Ownership / Investment / Acquisition / Subsidiary

For each sentence, extract:
- entity_from: The source of the product/service(Proper Noun ONLY. If none, null).
- entity_to: The receiver of the product/service (Proper Noun ONLY. If none, null).
- relation_type: A short string (e.g., "Supplier", "Partnership", "Ownership"). If relation_id is 0, this MUST be null.
- relation_id: Integer (0, 1, 2, 3, or 4) exactly matching the mapping above.
- confidence_score: A float between 0.0 and 1.0.
- reasoning: MAX 5 WORDS. Be extremely concise (e.g., "TSMC supplies chips to Apple").
- has_explicit_names: boolean, true ONLY if BOTH companies are explicitly named proper nouns.

Return ONLY a valid JSON object matching this structure:
{
  "results": [
    {
      "row_id": 123,
      "entity_from": "TSMC",
      "entity_to": "Apple",
      "relation_id": 1,
      "relation_type": "Supplier",
      "confidence_score": 0.95,
      "reasoning": "TSMC supplies chips to Apple",
      "has_explicit_names": true
    }
  ]
}
"""

def process_backlog():
    print("==============================================")
    print(f"   INITIATING {GEMINI_MODEL_NAME.upper()} (LIMIT: {MAX_LIMIT})")
    print("==============================================")
    
    total_processed = 0

    while total_processed < MAX_LIMIT:
        remaining_to_fetch = MAX_LIMIT - total_processed
        current_batch_size = min(BATCH_SIZE, remaining_to_fetch)
        
        res = supabase.table("scraped_sentences")\
            .select("*")\
            .eq("llm_processed", "not_started")\
            .limit(current_batch_size)\
            .execute()
            
        rows = res.data if hasattr(res, "data") else []
        
        if not rows:
            print("\n[✓] Backlog complete! No more 'not_started' rows in the database.")
            break
            
        print(f"\n[i] Fetched batch of {len(rows)} sentences. Sending to {GEMINI_MODEL_NAME}...")
        
        formatted_inputs = [{"row_id": r["id"], "source_company": r["source_company"], "sentence": r["raw_sentence"]} for r in rows]
        user_prompt = f"{SYSTEM_PROMPT}\n\nHere is the batch:\n{json.dumps({'batch': formatted_inputs})}"
        
        original_map = {r["id"]: r for r in rows}
        
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            parsed_response = json.loads(response.text)
            results_array = parsed_response.get("results", [])
            
            bulk_updates = []
            for item in results_array:
                row_id = item.get("row_id")
                if row_id not in original_map:
                    continue
                    
                orig_row = original_map[row_id]
                has_names = item.get("has_explicit_names", True)
                rel_id = item.get("relation_id", 0)
                
                status = "sus" if (not has_names and rel_id > 0) else "completed"
                
                payload = {
                    **orig_row, 
                    "entity_from": item.get("entity_from"),
                    "entity_to": item.get("entity_to"),
                    "relation_id": int(rel_id) if rel_id is not None else 0,
                    "relation_type": str(item.get("relation_type", "Unknown")),
                    "confidence_score": float(item.get("confidence_score", 0.0)),
                    "reasoning": item.get("reasoning", ""),
                    "llm_processed": status
                }
                bulk_updates.append(payload)
                
            # --- THE FIX: Deduplicate the updates based on row_id ---
            unique_updates = {}
            for update in bulk_updates:
                if update["id"] not in unique_updates:
                    unique_updates[update["id"]] = update
            
            final_updates = list(unique_updates.values())
            # --------------------------------------------------------

            if final_updates:
                supabase.table("scraped_sentences").upsert(final_updates).execute()
                total_processed += len(final_updates)
                print(f"  [+] Successfully processed and updated {len(final_updates)} rows. (Total so far: {total_processed} / {MAX_LIMIT})")
            else:
                # Fallback if LLM returned empty or mismatched IDs to prevent infinite loop
                print("  [!] LLM returned no valid IDs. Marking batch as 'failed' to move on.")
                fail_updates = [{**r, "llm_processed": "failed"} for r in rows]
                supabase.table("scraped_sentences").upsert(fail_updates).execute()
            
        except Exception as e:
            print(f"  [✗] Error processing batch: {e}")
            print("  [!] Sleeping for 10 seconds before retrying...")
            time.sleep(10)
            continue
            
        time.sleep(2)
        
    if total_processed >= MAX_LIMIT:
        print(f"\n[✓] Hard limit of {MAX_LIMIT} reached. Shutting down backlog cruncher safely.")

if __name__ == "__main__":
    process_backlog()