import os
import json
import re
import time
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from supabase import Client, create_client
from functools import partial
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from wiki_resolver import get_wiki_data

from .tools import tool_fetch_next_sentence, tool_analysing, tool_writedb

class LabelingState(TypedDict):
    queue_rows: List[Dict[str, Any]]
    final_results: List[Dict[str, Any]]
    pipeline_status: str
    
@traceable(run_type="tool", name="Record Sentence Extraction")
def trace_single_sentence(row_id: int, relation_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Virtual Trace to log individual sentences in LangSmith without extra DB calls."""
    return payload

def ingest_node(state: LabelingState, supabase: Client) -> Dict[str, Any]:
    """
    JOB: Fetches raw rows from the queue. Respects the safe batch size.
    """
    print("\n--- Ingest Node: Querying database queue ---")
    current_batch_size = int(os.getenv("BATCH_SIZE", "30"))
    rows = tool_fetch_next_sentence(supabase, batch_size=current_batch_size)
    
    if not rows:
        print("[i] Ingest Node: No sentences remaining with status 'not_started'.")
        return {"queue_rows": [], "final_results": [], "pipeline_status": "complete"}
    
    print(f"Ingest Node: Successfully fetched {len(rows)} sentences.")
    return {
        "queue_rows": rows,
        "final_results": [],
        "pipeline_status": "in_progress"
    }

def analysing_node(state: LabelingState) -> Dict[str, Any]:
    """
    JOB: Chunks inputs, coordinates local LLM inferences, strips markdown wrapper
         artifacts, and guarantees safe database formats even during engine crashes.
    """
    rows = state.get("queue_rows", [])
    if not rows:
        return {"final_results": []}

    print(f"--- Analysis Node: Processing {len(rows)} rows ---")
    aggregated_outputs = []
    
    formatted_inputs = [
        {
            "row_id": r["id"],
            "source_company": r["source_company"],
            "sentence": r["raw_sentence"]
        } for r in rows
    ]
    
    user_prompt = f"Analyze this batch of target sentences:\n{json.dumps({'batch': formatted_inputs}, indent=2)}"
    
    try:
        print(f"[i] Dispatching batch request to model..")
        raw_response = tool_analysing(user_prompt)
        
        time.sleep(2)
        
        print(f"    [Raw Output Snippet]: {raw_response.strip()[:150]}...")
        
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            # Slice out ONLY the JSON part
            cleaned_response = raw_response[start_idx:end_idx + 1]
        else:
            raise ValueError("LLM returned text with no JSON brackets `{}` inside it.")

        parsed_response = json.loads(cleaned_response)
        results_array = parsed_response.get("results", [])

        if not results_array:
            raise KeyError("LLM payload parsed as JSON but lacks the 'results' key.")

        for item in results_array:
            has_names = item.get("has_explicit_names", True)
            rel_id = item.get("relation_id", 0)
            status = "sus" if (not has_names and rel_id > 0) else "completed"
                
            aggregated_outputs.append({
                "id": item["row_id"],
                "entity_from": item.get("entity_from"),
                "entity_to": item.get("entity_to"),
                "relation_id": int(rel_id) if rel_id is not None else 0,
                "relation_type": str(item.get("relation_type", "Unknown")),
                "confidence_score": float(item.get("confidence_score", 0.0)),
                "reasoning": item.get("reasoning"),
                "llm_processed": status
            })
            
    except Exception as e:
        print(f"[✗] Critical parsing error occurred: {str(e)}. Constructing safe recovery payloads.")
        for fail_row in rows:
            aggregated_outputs.append({
                "id": fail_row["id"],
                "entity_from": None,
                "entity_to": None,
                "relation_id": 0,
                "relation_type": "ERROR_FAILED",
                "confidence_score": 0.0,
                "reasoning": f"Pipeline failure: {str(e)[:500]}", 
                "llm_processed": "failed" 
            })

    return {"final_results": aggregated_outputs}

def db_writer_node(state: LabelingState, supabase: Client) -> Dict[str, Any]:
    results = state.get("final_results", [])
    queue_rows = state.get("queue_rows", []) 
    
    if not results:
        print("[i] DB Writer Node: Zero entries to update.")
        return {}
        
    print(f"--- DB Writer Node: Committing updates for {len(results)} rows ---")
    
    original_map = {r["id"]: r for r in queue_rows}
    bulk_payloads = []
    
    discovered_entities_map = {}
    
    for res in results:
        orig_row = original_map.get(res["id"], {})
        payload = {
            **orig_row, 
            "entity_from": res.get("entity_from"),
            "entity_to": res.get("entity_to"),
            "relation_id": res.get("relation_id"),
            "relation_type": res.get("relation_type"),
            "confidence_score": res.get("confidence_score"),
            "reasoning": res.get("reasoning"),
            "llm_processed": res.get("llm_processed", "completed")
        }
        bulk_payloads.append(payload)
        trace_single_sentence(res["id"], str(res.get("relation_type", "Unknown")), payload)
        
        if (res.get("relation_id") or 0) > 0:
            source_company = orig_row.get("source_company")
            
            for entity in [res.get("entity_from"), res.get("entity_to")]:
                if entity and isinstance(entity, str):
                    clean_ent = entity.strip()
                    invalid_names = ["unknown", "the company", "they", "we", "our company", "its", "it", "anonymous", "[anonymous]", "[ANONYMOUS]"]
                    
                    if len(clean_ent) > 2 and clean_ent.lower() not in invalid_names and "anonymous" not in clean_ent.lower() and "[" not in clean_ent:
                        if clean_ent.lower() not in (source_company.lower() if source_company else ""):
                            discovered_entities_map[clean_ent] = {
                                "source": source_company,
                                "root_seed": orig_row.get("root_seed")
                            }
    
    success = tool_writedb(supabase, bulk_payloads)
    
    if success:
        print(f"DB Writer Node: Batch successfully written and cleared.")
        if discovered_entities_map:
            print(f"    [Flywheel] Found {len(discovered_entities_map)} potential new relationships. Resolving identities...")
            
            queue_res = supabase.table("company_queue").select("company_name, legal_name, tier_level").execute()
            queue_data = queue_res.data if hasattr(queue_res, "data") else []
            
            existing_names_map = {} 
            for q in queue_data:
                tier = q.get("tier_level") or 0
                existing_names_map[q["company_name"].lower()] = tier
                if q.get("legal_name"):
                    existing_names_map[q["legal_name"].lower()] = tier
            
            companies_to_insert = []

            for raw_name, data in discovered_entities_map.items():
                
                source_company = data["source"]
                root_seed = data["root_seed"]
                if raw_name.lower() in existing_names_map:
                    
                    continue
                
                print(f"      - Querying Wiki for canonical identity of: {raw_name}")
                wiki_data = get_wiki_data(raw_name)
                
                if wiki_data and wiki_data.get("legal_name"):
                    canonical_name = wiki_data["legal_name"]
                    jurisdiction = wiki_data.get("jurisdiction_code")
                    wiki_url = wiki_data.get("opencorporates_id")
                else:
                    canonical_name = raw_name
                    jurisdiction = None
                    wiki_url = None
                
                if canonical_name.lower() in existing_names_map:
                    print(f"      - [Skipped] {raw_name} resolved to {canonical_name}, which we already have.")
                    continue
                
                source_tier = existing_names_map.get(str(source_company).lower(), 0)
                new_tier = source_tier + 1
                
                assign_status = "not_started" if jurisdiction else "gray_zone"
                
                companies_to_insert.append({
                    "company_name": canonical_name,
                    "legal_name": canonical_name,
                    "tier_level": new_tier,
                    "status": assign_status,
                    "root_seed": root_seed,
                    "jurisdiction_code": jurisdiction,
                    "opencorporates_id": wiki_url
                })
                
                # Add to our local map so we don't duplicate it in this same loop
                existing_names_map[canonical_name.lower()] = new_tier
                
            # C. Insert the globally resolved, tier-accurate companies!
            if companies_to_insert:
                try:
                    supabase.table("company_queue").insert(companies_to_insert).execute()
                    print(f"    [+] FLYWHEEL: Injected {len(companies_to_insert)} newly resolved companies into the queue!")
                except Exception as e:
                    print(f"    [!] Flywheel Error inserting: {e}")
            else:
                print("    [-] Flywheel: All discovered entities were already in the database.")
    else:
        print("DB Writer Node: Fatal database error during batch update.")
        
    return {}


def route_after_ingestion(state: LabelingState) -> str:
    """
    Evaluates state to safely close graph execution loops.
    """
    if not state.get("queue_rows") or len(state["queue_rows"]) == 0:
        return "end"
    return "analyze"

def build_pipeline_graph(supabase_client: Client):
    """
    Compiles state workflow map
    """
    workflow = StateGraph(LabelingState)
    
    bound_ingest = partial(ingest_node, supabase=supabase_client)
    bound_writer = partial(db_writer_node, supabase=supabase_client)
    
    workflow.add_node("ingest", bound_ingest)
    workflow.add_node("analyze", analysing_node)
    workflow.add_node("write_db", bound_writer)
    
    workflow.add_edge(START, "ingest")
    
    workflow.add_conditional_edges(
        "ingest",
        route_after_ingestion,
        {
            "analyze": "analyze", 
            "end": END
        }
    )
    workflow.add_edge("analyze", "write_db")
    workflow.add_edge("write_db", "ingest")
    
    return workflow.compile()

def run_pipeline_batch():
    """
    Entry point called by the CLI (run.py). 
    Initializes the database and triggers the continuous LangGraph loop.
    """
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("[CRITICAL] Missing Supabase credentials for the pipeline.")
        return
        
    supabase_client = create_client(url, key)
    
    # 1. Compile the graph
    app = build_pipeline_graph(supabase_client)
    
    print("\n[i] LangGraph Engine Started. Draining the 'not_started' queue...")
    
    final_state = app.invoke({
        "queue_rows": [],
        "final_results": [],
        "pipeline_status": "starting"
    })
    
    print(f"\n[✓] LangGraph Engine Shutdown. Queue is fully drained.")
    return final_state