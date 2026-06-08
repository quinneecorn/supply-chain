import os
import sys
sys.path.append(os.path.abspath("bilstm"))
import time
import re
import threading
import math
import tkinter as tk
from queue import Queue
from dotenv import load_dotenv
from supabase import create_client, Client
from bilstm.gliner_en import process_with_gliner
import torch
from bilstm.model import BiLSTMClassifier
from transformers import BertTokenizer

print("[i] Loading BiLSTM Model and Tokenizer into memory...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 1. Add the custom mask tokens!
tokenizer.add_special_tokens({'additional_special_tokens': ['[__NE_FROM__]', '[__NE_TO__]']})

# 2. Initialize with exact dimensions
model = BiLSTMClassifier(
    vocab_size=len(tokenizer),  
    embed_dim=256,              
    hidden_dim=384,             
    num_classes=4               
)

# 3. Unpack and load the specific weights
checkpoint = torch.load('bilstm/bilstm_relation.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[CRITICAL] Missing Supabase credentials in .env")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 1. THE POP-UP WINDOW GUI 
# ==========================================
def gui_worker(update_queue: Queue, db_client: Client):
    root = tk.Tk()
    root.title("Live BiLSTM Extraction Feed Dashboard")
    root.geometry("800x800")
    root.attributes('-topmost', True) 
    root.configure(bg='#1e1e1e')
    
    # --- Top Bar ---
    btn_frame = tk.Frame(root, bg='#1e1e1e')
    btn_frame.pack(fill='x', padx=10, pady=5)

    tk.Label(btn_frame, text="Seed:", fg="white", bg="#1e1e1e").pack(side='left', padx=(5, 2))
    seed_entry = tk.Entry(btn_frame, width=15)
    seed_entry.pack(side='left', padx=5)

    tk.Label(btn_frame, text="Tier:", fg="white", bg="#1e1e1e").pack(side='left', padx=(5, 2))
    tier_entry = tk.Entry(btn_frame, width=5)
    tier_entry.pack(side='left', padx=5)

    text_area = tk.Text(root, height=6, state='disabled', bg='#000000', fg='#00ff00', font=('Courier', 10))
    def log_text(msg: str):
        text_area.config(state='normal')
        text_area.insert(tk.END, msg + "\n")
        text_area.config(state='disabled')
        text_area.see(tk.END)

    categorized_data = { 1: set(), 2: set(), 3: set(), 4: set() }

    def start_pipeline():
        seed = seed_entry.get().strip()
        tier_str = tier_entry.get().strip()
        if not seed or not tier_str:
            log_text("[!] Please enter both a seed and a tier.")
            return
        try:
            tier = int(tier_str)
            for k in categorized_data: categorized_data[k].clear()
            redraw_lists()
            threading.Thread(target=run_nlp_pipeline, args=(seed, tier, update_queue), daemon=True).start()
        except ValueError:
            log_text("[!] Target Tier must be a number.")

    tk.Button(btn_frame, text="▶ Run BiLSTM", command=start_pipeline, bg="#007bff", fg="black").pack(side='left', padx=10)

    # --- Gray Zone Functions ---
    def show_gray_zone():
        try:
            res = db_client.table("company_queue").select("*").eq("status", "gray_zone").execute()
            comps = res.data if hasattr(res, "data") else (res if isinstance(res, list) else [])
            
            view_win = tk.Toplevel(root)
            view_win.title("Gray Zone Queue")
            view_win.geometry("400x300")
            view_win.attributes('-topmost', True)
            view_win.configure(bg='#1e1e1e')
            
            tk.Label(view_win, text="Companies Pending Review:", fg="white", bg="#1e1e1e", font=('Arial', 12, 'bold')).pack(pady=10)
            listbox = tk.Listbox(view_win, bg='#0f0f0f', fg='#ffffff', font=('Arial', 11))
            listbox.pack(padx=20, pady=5, fill='both', expand=True)
            
            if not comps:
                listbox.insert(tk.END, "✅ Empty! No companies in the Gray Zone.")
                listbox.itemconfig(0, {'fg': '#00ff00'})
            else:
                for c in comps:
                    listbox.insert(tk.END, f" Tier {c.get('tier_level', '?')} | {c.get('company_name', 'Unknown')}")
        except Exception as e:
            log_text(f"[!] Error loading Gray Zone: {e}")

    def run_gray_zone():
        res = db_client.table("company_queue").select("*").eq("status", "gray_zone").execute()
        comps = res.data if hasattr(res, "data") else []
        if not comps: return
            
        handler_win = tk.Toplevel(root)
        handler_win.title("Gray Zone Management")
        handler_win.attributes('-topmost', True)
        listbox = tk.Listbox(handler_win, width=50, height=10)
        listbox.pack(padx=20, pady=5)
        for c in comps: listbox.insert(tk.END, f"{c['id']} | {c['company_name']}")
            
        def approve():
            sel = listbox.curselection()
            if sel:
                comp = comps[sel[0]]
                db_client.table("company_queue").update({"status": "not_started"}).eq("id", comp['id']).execute()
                log_text(f"[+] Approved {comp['company_name']}.")
                listbox.delete(sel[0]); del comps[sel[0]]
                
        def delete_it():
            sel = listbox.curselection()
            if sel:
                comp = comps[sel[0]]
                db_client.table("company_queue").delete().eq("id", comp['id']).execute()
                log_text(f"[-] Deleted {comp['company_name']}.")
                listbox.delete(sel[0]); del comps[sel[0]]

        tk.Button(handler_win, text="Approve", command=approve, bg="#28a745").pack(side="left", padx=20, pady=10)
        tk.Button(handler_win, text="Delete", command=delete_it, bg="#dc3545").pack(side="right", padx=20, pady=10)

    tk.Button(btn_frame, text="🔍 Show Gray Zone", command=show_gray_zone).pack(side='left', padx=5)
    tk.Button(btn_frame, text="⚙️ Run Gray Zone", command=run_gray_zone).pack(side='left', padx=5)

    # --- Dashboard Area ---
    frame = tk.Frame(root, bg='#1e1e1e')
    frame.pack(expand=True, fill='both', padx=10, pady=5)
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side='right', fill='y')
    
    display_area = tk.Text(frame, bg='#0f0f0f', fg='#ffffff', font=('Arial', 12), yscrollcommand=scrollbar.set, state='disabled')
    display_area.pack(expand=True, fill='both')
    scrollbar.config(command=display_area.yview)
    text_area.pack(fill='x', padx=10, pady=5)
    
    def redraw_lists():
        display_area.config(state='normal')
        display_area.delete(1.0, tk.END)
        
        supply_lines = []
        for src, tgt in categorized_data[1]: supply_lines.append(f" • {tgt} -> {src}\n")
        for src, tgt in categorized_data[2]: supply_lines.append(f" • {src} -> {tgt}\n")
            
        display_area.insert(tk.END, "=== 🚚 SUPPLY CHAIN ===\n", "head_1")
        for line in sorted(list(set(supply_lines))): display_area.insert(tk.END, line)
        display_area.insert(tk.END, "\n")
        
        display_area.insert(tk.END, "=== 🤝 PARTNERSHIP ===\n", "head_3")
        for src, tgt in sorted(categorized_data[3]): display_area.insert(tk.END, f" • {src} <-> {tgt}\n")
        display_area.insert(tk.END, "\n")
        
        display_area.insert(tk.END, "=== 👑 OWNERSHIP ===\n", "head_4")
        for src, tgt in sorted(categorized_data[4]): display_area.insert(tk.END, f" • {src} owns {tgt}\n")
        
        display_area.tag_config("head_1", foreground="#00ffff", font=('Arial', 12, 'bold'))
        display_area.tag_config("head_3", foreground="#00ff00", font=('Arial', 12, 'bold'))
        display_area.tag_config("head_4", foreground="#ff9900", font=('Arial', 12, 'bold'))
        display_area.config(state='disabled')

    def check_queue():
        needs_redraw = False
        while not update_queue.empty():
            data = update_queue.get()
            
            if isinstance(data, str):
                log_text(data)
            elif isinstance(data, dict) and data["type"] == "edge":
                current_tier_str = tier_entry.get().strip()
                current_tier = int(current_tier_str) if current_tier_str.isdigit() else 1
                
                if data.get("target_tier", 1) <= current_tier:
                    rel = data["relation"]
                    src, tgt = data["source"].strip(), data["target"].strip()

                    if src.lower() != tgt.lower() and len(src) > 2 and len(tgt) > 2:
                        bad_words = ["the company", "inc", "inc.", "ltd", "ltd.", "corp", "corp.", "the committee", "board of directors"]
                        if src.lower() not in bad_words and tgt.lower() not in bad_words:
                            if rel in categorized_data:
                                categorized_data[rel].add((src, tgt))
                                needs_redraw = True

        if needs_redraw: redraw_lists()
        root.after(50, check_queue)

    root.after(100, check_queue)
    root.mainloop()

# ==========================================
# 2. CORE UTILITIES & PIPELINE
# ==========================================
def patch_broken_root_seeds(seed_company: str):
    """[DUMMY FUNCTION] Retroactively tags rows mentioning the seed_company with the correct root_seed."""
    try:
        res = supabase.table("scraped_sentences") \
            .update({"root_seed": seed_company}) \
            .ilike("raw_sentence", f"%{seed_company}%") \
            .neq("root_seed", seed_company) \
            .execute()
        
        fixed_count = len(res.data) if hasattr(res, "data") and res.data else 0
        if fixed_count > 0:
            print(f"  [+] Retroactively fixed {fixed_count} orphaned rows for '{seed_company}'!")
    except Exception as e:
        print(f"  [!] Patch error: {e}")

def run_nlp_pipeline(seed_company: str, target_tier: int, ui_queue: Queue):
    """Pulls sentences from DB, masks them with GLiNER, and runs PyTorch inference."""
    print(f"\n--- INIT: BiLSTM Pipeline starting for {seed_company} (Targeting Tier {target_tier}) ---")
    
    # 1. DUMMY TRACE: Fix broken root seeds right now so we don't miss history
    patch_broken_root_seeds(seed_company)

    # 2. RESUME FIX: Thaw any stuck sentences from previous cancelled runs
    print(f"[i] Checking for stuck 'in_progress' sentences to resume...")
    supabase.table("scraped_sentences").update({"llm_processed": "not_started"}).eq("llm_processed", "in_progress").ilike("root_seed", seed_company).execute()

    # 3. THE WHITELIST: Fetch the clean whitelist from company_queue
    print(f"[i] Fetching clean whitelist from company_queue for Tier <= {target_tier}...")
    queue_res = supabase.table("company_queue").select("company_name").ilike("root_seed", seed_company).lte("tier_level", target_tier).execute()
    valid_companies = {row['company_name'].lower() for row in queue_res.data} if hasattr(queue_res, "data") and queue_res.data else set()
    valid_companies.add(seed_company.lower()) # Always ensure the seed is valid

    # 4. INSTANT UI FEED: Show ALREADY EXISTING relationships
    print(f"[i] Checking database for historical relationships...")
    history_res = supabase.table("scraped_sentences").select("*").eq("llm_processed", "completed").ilike("root_seed", seed_company).in_("relation_id", [1, 2, 3]).execute()
    
    historical_rows = history_res.data if hasattr(history_res, "data") else []
    if historical_rows:
        clean_count = 0
        for row in historical_rows:
            src = row['entity_from']
            tgt = row['entity_to']
            
            # STRICT FILTER: Both companies MUST exist in your validated queue
            if src and tgt and src.lower() in valid_companies and tgt.lower() in valid_companies:
                ui_queue.put({
                    "type": "edge", 
                    "source": src, 
                    "target": tgt, 
                    "source_tier": target_tier - 1, 
                    "target_tier": target_tier,
                    "relation": row['relation_id'],
                    "tag": "HIST"
                })
                clean_count += 1
                time.sleep(0.01) 
        print(f"[i] Displayed {clean_count} strict existing extractions instantly.")
    else:
        print("[i] No existing relationships found.")

    # 5. NEW BATCH PROCESSING: Query the database for unprocessed text
    print("\n[i] Kicking off BiLSTM batch processing to find new data...")
    res = supabase.table("scraped_sentences").select("*").eq("llm_processed", "not_started").ilike("root_seed", seed_company).execute()
    pending_sentences = res.data if hasattr(res, "data") else []

    if not pending_sentences:
        print("[i] No pending sentences found for this seed.")
        return

    print(f"[i] Found {len(pending_sentences)} sentences to process.")

    # Process each row interactively
    for row in pending_sentences:
        row_id = row['id']
        raw_text = row['raw_sentence']

        # Lock the row
        supabase.table("scraped_sentences").update({"llm_processed": "in_progress"}).eq("id", row_id).execute()

        # GLiNER Masking
        gliner_result = process_with_gliner(raw_text)
        
        if not gliner_result or not gliner_result.get("entity_from") or not gliner_result.get("entity_to"):
            supabase.table("scraped_sentences").update({"llm_processed": "failed"}).eq("id", row_id).execute()
            continue

        masked_text = gliner_result["masked_sentence"]
        e_from = gliner_result["entity_from"]
        e_to = gliner_result["entity_to"]
        
        # PyTorch Inference
        try:
            inputs = tokenizer(
                masked_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            
            with torch.no_grad():
                logits = model(inputs['input_ids']) 
                
            predicted_class = torch.argmax(logits, dim=1).item()
        except Exception as e:
            print(f"  [!] PyTorch Error on Row {row_id}: {e}")
            predicted_class = 0

        # Save and Dispatch
        if predicted_class in [1, 2, 3]: 
            supabase.table("scraped_sentences").update({
                "llm_processed": "completed",
                "masked_sentence": masked_text,
                "entity_from": e_from,
                "entity_to": e_to,
                "relation_id": predicted_class
            }).eq("id", row_id).execute()
            
            # STRICT FILTER
            if e_from.lower() in valid_companies and e_to.lower() in valid_companies:
                ui_queue.put({
                    "type": "edge", 
                    "source": e_from, 
                    "target": e_to, 
                    "source_tier": target_tier - 1, 
                    "target_tier": target_tier,
                    "relation": predicted_class,
                    "tag": "LIVE"
                })
                print(f"  [✓] Row {row_id} | Saved & Displayed (Type {predicted_class})")
            else:
                print(f"  [~] Row {row_id} | Saved but HIDDEN (Entities not strictly whitelisted)")
        else:
            supabase.table("scraped_sentences").update({
                "llm_processed": "completed",
                "masked_sentence": masked_text,
                "entity_from": e_from,
                "entity_to": e_to,
                "relation_id": 0
            }).eq("id", row_id).execute()

    print("\n--- NLP Batch Processing Complete ---")

# ==========================================
# 3. MAIN THREAD EXECUTION
# ==========================================
def main_loop(ui_queue: Queue):
    print(r"""
    ========================================
       BiLSTM Relation Extraction Engine
    ========================================
    """)
    
    while True:
        target = input("\nEnter a Root Seed Company (or 'quit' to exit): ").strip()
        
        if target.lower() in ['q', 'quit', 'exit']:
            break
        if not target:
            continue
            
        try:
            target_tier = int(input(f"Enter the Target Tier Level for {target} (e.g., 1): ").strip())
        except ValueError:
            print("[!] Invalid tier level. Please enter a number.")
            continue
            
        run_nlp_pipeline(seed_company=target, target_tier=target_tier, ui_queue=ui_queue)

if __name__ == "__main__":
    live_feed_queue = Queue()
    pipeline_thread = threading.Thread(target=main_loop, args=(live_feed_queue,), daemon=True)
    pipeline_thread.start()
    time.sleep(0.5)

    try:
        gui_worker(live_feed_queue, supabase)
    except KeyboardInterrupt:
        print("\n[!] Pipeline stopped by user.")