import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
from gemma.pipeline import build_pipeline_graph

def main():
    load_dotenv()
    
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("LangSmith Tracing: ENABLED.")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Critical Error: Missing Supabase credentials in .env file.")
        sys.exit(1)
        
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("[✓] Connected securely to Supabase.")
    except Exception as e:
        print(f"[✗] Failed to connect to Supabase: {str(e)}")
        sys.exit(1)

    print("Compiling LangGraph pipeline...")
    app = build_pipeline_graph(supabase)
    
    print("=" * 50)
    print("Starting Data Factory Queue Processor...")
    print("=" * 50)

    batch_cycle = 0
    total_processed_rows = 0

    # 4. The Processing Loop
    while True:
        batch_cycle += 1
        print(f"\nStarting Batch Cycle {batch_cycle}...")
        
        initial_state = {
            "queue_rows": [],
            "final_results": [],
            "pipeline_status": "not_started"
        }
        
        final_output = app.invoke(initial_state)
        
        status = final_output.get("pipeline_status", "")
        if status == "complete":
            print("\n" + "=" * 50)
            print("Execution Finished! No more 'not_started' rows found.")
            print(f"Total batches processed: {batch_cycle - 1}")
            print(f"Total sentences labeled: {total_processed_rows}")
            print("=" * 50)
            break

        # Calculate metrics for the current run
        current_batch_count = len(final_output.get("queue_rows", []))
        total_processed_rows += current_batch_count
        
        print(f"✓ Cycle {batch_cycle} Complete. Cumulative rows processed: {total_processed_rows}")

if __name__ == "__main__":
    main()