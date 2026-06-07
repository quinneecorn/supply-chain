# Gemini NER & Masking Pipeline

This folder contains the LLM-powered pipeline for high-quality Named Entity Recognition (NER) and masking.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Update the `CONFIG` in `NER_Pipeline.py` and `reset_db.py` with your Supabase and OpenRouter keys.

## Scripts
- **`NER_Pipeline.py`**: The main processing engine. Uses multi-streaming (10 workers) to process records in parallel.
- **`reset_db.py`**: Wipes the masking results in the `NLP_copy_test` table and resets status to `not_started`.
- **`Discovery_Schema.py`**: Helper tool to list tables and columns in your project.

## Run
```bash
python NER_Pipeline.py
```
The script will loop until all records are processed.
