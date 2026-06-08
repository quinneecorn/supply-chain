import os
from typing import List, Dict, Any
from supabase import Client
from openai import OpenAI
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ai_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
# GEMINI_MODEL_NAME = "gemini-3.5-flash"
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

ANALYSIS_SYSTEM_PROMPT = """
You are an expert supply chain data analyst extracting structured relationships from SEC corporate filings.
You will be given a JSON array of corporate sentences. Analyze every sentence independently. 
You must return a strict JSON object containing a top-level "results" array.
Every single item inside your output array MUST map back to its original input "row_id" so data lineage is preserved.

ABSOLUTE ZERO-TOLERANCE RULES FOR ENTITIES (NO GHOST ENTITIES):
1. PROPER NOUNS ONLY: `entity_from` and `entity_to` MUST be specific, real-world named companies (e.g., "Apple", "Foxconn", "TSMC").
2. FORBIDDEN WORDS: You are strictly forbidden from using generic nouns, pronouns, or roles (e.g., "banks", "the supplier", "customers", "we", "the company", "our partners", "third-party").
3. THE NULL RULE: If a specific company name is missing for either the buyer or the supplier, you MUST set that field to null.
4. If a relationship exists (IDs 1-4) but one or both company names are hidden or anonymous, set 'has_explicit_names' to false.
5. Provide a placeholder description for the hidden entity (e.g., "[ANONYMOUS] supplier").

You must classify the sentence into exactly ONE of these relation IDs:
0 - No relationship / Reject
1 - B supplies A (Entity B provides goods to Entity A).
2 - A supplies B (Entity A provides goods to Entity B).
3 - Partnership, Collaboration (Strategic alliances/joint ventures).
4 - Ownership (Parent/subsidiary/equity stake, acquisition).

For each sentence, extract:
- entity_from: The source of the product/service(Proper Noun ONLY. If none, null).
- entity_to: The receiver of the product/service (Proper Noun ONLY. If none, null).
- relation_type: A short string (e.g., "Supplier", "Partnership", "Ownership").
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

def tool_fetch_next_sentence(supabase: Client, batch_size: int = int(os.getenv("BATCH_SIZE"))) -> List[Dict[str, Any]]:
    """
    JOB: Pure read operation. Dynamically fetches the exact batch size requested.
    """
    try:
        response = (
            supabase.table("scraped_sentences")
            .select("*") 
            .eq("llm_processed", "not_started")
            .limit(batch_size) 
            .execute()
        )
        return response.data if response.data else []
    except Exception as e:
        print(f"[!] DB Fetch Error: {str(e)}")
        return []

def tool_analysing(user_prompt: str) -> str:
    """
    JOB: Pure API call. Returns raw text. 
    JSON parsing and Markdown stripping is strictly handled by the pipeline.
    """
    
    try:
        # response = ai_client.chat.completions.create(
        #     model="gemma",
        #     messages=[
        #         {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT.strip()},
        #         {"role": "user", "content": user_prompt}
        #     ],
        #     temperature=0.1,
        #     response_format={"type": "json_object"}
        # )
        response = gemini_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=ANALYSIS_SYSTEM_PROMPT.strip(),
                    response_mime_type="application/json",
                )
            )
        #return response.choices[0].message.content #gemma
        return response.text #gemini
    except Exception as e:
        raise RuntimeError(f"LLM API Call Failed: {str(e)}")

def tool_writedb(supabase: Client, payload_list: List[Dict[str, Any]]) -> bool:
    """
    JOB: Pure write operation. Upserts the completely formatted list of dictionaries.
    """
    if not payload_list:
        return True
    try:
        supabase.table("scraped_sentences").upsert(payload_list).execute()
        return True
    except Exception as e:
        print(f"[!] DB Write Error: {str(e)}")
        return False