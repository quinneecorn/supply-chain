import os
import sys
import time
import threading
import tkinter as tk
from queue import Queue
from dotenv import load_dotenv
from supabase import create_client, Client

# Crawlers and Resolvers
from wiki_resolver import get_wiki_data
from crawler.sec_crawler import SECCrawler
from crawler.chs_crawler import CHSCrawler
from crawler.newsapi_crawler import NewsCrawler

# The LLM Pipeline
from gemma.pipeline import run_pipeline_batch 

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[CRITICAL] Missing Supabase credentials in .env")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 1. CORE LLM & CRAWLER LOGIC
# ==========================================
def inject_seed_company(company_name: str, ui_queue: Queue = None) -> dict:
    """Checks if a company exists. If not, resolves its identity. If incomplete, resets it to resume."""
    msg = f"[*] Checking database for '{company_name}'..."
    print(msg); ui_queue.put(msg) if ui_queue else None
    
    res = supabase.table("company_queue").select("*").ilike("company_name", company_name).execute()
    if hasattr(res, "data") and res.data:
        comp = res.data[0]
        
        # --- THE RESUME FIX ---
        # If it exists but didn't finish, reset it so the BFS crawler picks it up!
        if comp.get('status') in ['in_progress', 'half_processed', 'failed']:
            msg = f"  [!] '{comp['company_name']}' was stuck as '{comp.get('status')}'. Resetting to resume scraping!"
            print(msg); ui_queue.put(msg) if ui_queue else None
            
            supabase.table("company_queue").update({"status": "not_started"}).eq("id", comp['id']).execute()
            comp['status'] = 'not_started' # Update our local copy so the pipeline knows
        else:
            msg = f"  [i] '{comp['company_name']}' is already in the database as '{comp.get('status')}'."
            print(msg); ui_queue.put(msg) if ui_queue else None
            
        return comp
        
    msg = f"  [i] New company. Resolving global identity via Wikipedia..."
    print(msg); ui_queue.put(msg) if ui_queue else None
    
    wiki_data = get_wiki_data(company_name)
    if wiki_data and wiki_data.get("legal_name"):
        canonical_name = wiki_data["legal_name"]
        jurisdiction = wiki_data.get("jurisdiction_code")
        wiki_url = wiki_data.get("opencorporates_id")
        msg = f"  [✓] Identity resolved: {canonical_name}"
        print(msg); ui_queue.put(msg) if ui_queue else None
    else:
        canonical_name = company_name
        jurisdiction = None
        wiki_url = None

    res_canon = supabase.table("company_queue").select("*").ilike("company_name", canonical_name).execute()
    if not res_canon.data:
        res_canon = supabase.table("company_queue").select("*").ilike("legal_name", canonical_name).execute()

    if hasattr(res_canon, "data") and res_canon.data:
         return res_canon.data[0]

    assign_status = "not_started" if jurisdiction else "gray_zone"
    
    payload = {
        "company_name": canonical_name,
        "legal_name": canonical_name,
        "tier_level": 0,
        "status": assign_status,
        "root_seed": canonical_name,
        "jurisdiction_code": jurisdiction,
        "opencorporates_id": wiki_url
    }
    
    insert_res = supabase.table("company_queue").insert(payload).execute()
    msg = f"  [+] Injected {canonical_name} as a Tier 0 Seed!"
    print(msg); ui_queue.put(msg) if ui_queue else None
    return insert_res.data[0]

def run_crawlers_for_company(company_data: dict, ui_queue: Queue):
    company_name = company_data["company_name"]
    jurisdiction = company_data.get("jurisdiction_code")
    uk_crn = company_data.get("uk_crn")
    root_seed = company_data.get("root_seed")
    company_id = company_data["id"]
    
    msg = f"  >>> Scraping filings for: {company_name}"
    print(msg); ui_queue.put(msg)
    supabase.table("company_queue").update({"status": "in_progress"}).eq("id", company_data["id"]).execute()
    
    NewsCrawler(supabase).crawl(company_name, root_seed=root_seed)
    
    if jurisdiction == "gb":
        CHSCrawler(supabase).crawl(company_name, uk_crn=uk_crn, root_seed=root_seed, company_id=company_id)
    else:
        SECCrawler(supabase).crawl(company_name, root_seed=root_seed, company_id=company_id)
        
    supabase.table("company_queue").update({"status": "completed"}).eq("id", company_data["id"]).execute()

def run_bfs_trace(seed_company: str, max_scrape_tier: int, ui_queue: Queue):
    ui_queue.put(f"\n[i] INITIATING BREADTH-FIRST SEARCH (UP TO TIER {max_scrape_tier})")
    
    for current_tier in range(max_scrape_tier + 1):
        ui_queue.put(f"\n=== BFS LEVEL: PROCESSING TIER {current_tier} ===")
    
        while True:
            res = supabase.table("company_queue").select("*").eq("status", "not_started").eq("tier_level", current_tier).execute()
            pending_companies = res.data if hasattr(res, "data") else []
            
            if not pending_companies:
                ui_queue.put(f"[i] No pending companies at Tier {current_tier} to scrape.")
                break
                
            ui_queue.put(f"[i] Found {len(pending_companies)} companies to scrape.")
            for company in pending_companies:
                run_crawlers_for_company(company, ui_queue)
        
        ui_queue.put(f"[i] Scraping complete. Applying SQL Trace to isolate '{seed_company}'...")
        
        try:
            # --- THE SQL TRACE DUMMY HACK ---
            # 1. Freeze the entire 23k backlog
            supabase.table("scraped_sentences").update({"llm_processed": "on_hold"}).eq("llm_processed", "not_started").execute()
            
            # 2. Thaw ONLY the sentences containing our exact company name (bypassing broken root_seed)
            supabase.table("scraped_sentences").update({"llm_processed": "not_started"}).eq("llm_processed", "on_hold").ilike("raw_sentence", f"%{seed_company}%").execute()
            
            # 3. Fire the LLM Batch. It is now trapped in a room with ONLY the targeted sentences!
            ui_queue.put(f"[i] Waking up LLM Pipeline for focused extraction...")
            run_pipeline_batch() 
            
            # 4. Thaw the rest of the backlog back to normal
            supabase.table("scraped_sentences").update({"llm_processed": "not_started"}).eq("llm_processed", "on_hold").execute()
            # ---------------------------------
            
            ui_queue.put(f"[✓] Targeted Extraction for Tier {current_tier} Complete!")
        except Exception as e:
            ui_queue.put(f"[!] Pipeline error: {e}")
            # Safety net: ensure backlog is thawed even if LLM crashes
            supabase.table("scraped_sentences").update({"llm_processed": "not_started"}).eq("llm_processed", "on_hold").execute()
            
    ui_queue.put(f"\n[✓] BFS TRACE COMPLETE")
    
def patch_broken_root_seeds(seed_company: str, ui_queue: Queue):
    """
    [DUMMY FUNCTION - DELETE LATER WHEN CRAWLER ROOT INGESTION IS FIXED]
    Scans scraped_sentences and retroactively tags any row mentioning 
    the seed_company with the correct root_seed.
    """
    ui_queue.put(f"[!] DUMMY TRACE: Patching broken root_seeds for '{seed_company}'...")
    try:
        # Hijack any row containing the name that doesn't have the correct root_seed
        res = supabase.table("scraped_sentences") \
            .update({"root_seed": seed_company}) \
            .ilike("raw_sentence", f"%{seed_company}%") \
            .neq("root_seed", seed_company) \
            .execute()
        
        fixed_count = len(res.data) if hasattr(res, "data") and res.data else 0
        if fixed_count > 0:
            ui_queue.put(f"  [+] Retroactively fixed {fixed_count} orphaned rows!")
    except Exception as e:
        ui_queue.put(f"  [!] Patch error: {e}")

# ==========================================
# 2. THE THREAD ORCHESTRATOR
# ==========================================
def execute_llm_pipeline(seed_company: str, target_tier: int, ui_queue: Queue):
    """Orchestrates the crawling, LLM extraction, and final UI rendering."""
    
    # 1. Inject the Seed Company First
    inject_seed_company(seed_company, ui_queue)
    
    # ---> INJECT DUMMY TRACE HERE: Fix historical rows before we fetch them!
    patch_broken_root_seeds(seed_company, ui_queue)

    # 2. THE INITIAL WHITELIST: Fetch the clean list of companies right now
    ui_queue.put(f"\n[i] Fetching clean whitelist from company_queue for Tier <= {target_tier}...")
    queue_res = supabase.table("company_queue").select("company_name").ilike("root_seed", seed_company).lte("tier_level", target_tier).execute()
    valid_companies = {row['company_name'].lower() for row in queue_res.data} if hasattr(queue_res, "data") and queue_res.data else set()

    # 3. PUSH EXISTING TO DASHBOARD FIRST: Check database for historical relationships
    ui_queue.put(f"[i] Checking database for ALREADY EXISTING relationships...")
    history_res = supabase.table("scraped_sentences").select("*").eq("llm_processed", "completed").ilike("root_seed", seed_company).in_("relation_id", [1, 2, 3]).execute()
    
    historical_rows = history_res.data if hasattr(history_res, "data") else []
    if historical_rows:
        clean_count = 0
        for row in historical_rows:
            src = row.get('entity_from')
            tgt = row.get('entity_to')
            
            # STRICT FILTER
            if src and tgt and src.lower() in valid_companies and tgt.lower() in valid_companies:
                ui_queue.put({
                    "type": "edge", 
                    "source": src, 
                    "target": tgt, 
                    "target_tier": target_tier,
                    "relation": row['relation_id']
                })
                clean_count += 1
                time.sleep(0.02) 
        ui_queue.put(f"[✓] Displayed {clean_count} existing extractions on the dashboard.")
    else:
        ui_queue.put("[i] No existing relationships found.")

    # 4. RUN NEW CRAWLS & LLM BATCH
    ui_queue.put(f"\n[i] Kicking off background crawlers and targeted LLM...")
    run_bfs_trace(seed_company=seed_company, max_scrape_tier=target_tier, ui_queue=ui_queue)
    
    # 5. RE-SYNC DASHBOARD AFTER LLM FINISHES
    ui_queue.put(f"\n[i] Re-syncing dashboard with newly discovered relationships...")
    
    # Update Whitelist (since BFS might have added new companies)
    queue_res_new = supabase.table("company_queue").select("company_name").ilike("root_seed", seed_company).lte("tier_level", target_tier).execute()
    valid_companies_new = {row['company_name'].lower() for row in queue_res_new.data} if hasattr(queue_res_new, "data") and queue_res_new.data else set()
    
    # Update History (pulling the fresh batch)
    history_res_new = supabase.table("scraped_sentences").select("*").eq("llm_processed", "completed").ilike("root_seed", seed_company).in_("relation_id", [1, 2, 3]).execute()
    new_historical_rows = history_res_new.data if hasattr(history_res_new, "data") else []
    
    if new_historical_rows:
        for row in new_historical_rows:
            src = row.get('entity_from')
            tgt = row.get('entity_to')
            
            if src and tgt and src.lower() in valid_companies_new and tgt.lower() in valid_companies_new:
                ui_queue.put({
                    "type": "edge", 
                    "source": src, 
                    "target": tgt, 
                    "target_tier": target_tier,
                    "relation": row['relation_id']
                })
        ui_queue.put(f"[✓] Dashboard fully updated with all new relationships!")
        

# ==========================================
# 3. THE TKINTER DASHBOARD
# ==========================================
def gui_worker(update_queue: Queue, db_client: Client):
    root = tk.Tk()
    root.title("Live Extraction Feed Dashboard")
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
            threading.Thread(target=execute_llm_pipeline, args=(seed, tier, update_queue), daemon=True).start()
        except ValueError:
            log_text("[!] Target Tier must be a number.")

    tk.Button(btn_frame, text="▶ Run Batch", command=start_pipeline, bg="#007bff", fg="black").pack(side='left', padx=10)

    # --- Gray Zone ---
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

    # --- Dashboard UI ---
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

                    # Garbage Filter
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

if __name__ == "__main__":
    live_feed_queue = Queue()
    try:
        gui_worker(live_feed_queue, supabase)
    except KeyboardInterrupt:
        print("\n[!] Pipeline stopped.")