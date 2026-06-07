# GLiNER Local NER & Masking Pipeline

This folder contains the local processing pipeline using the zero-shot GLiNER model.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Update the `CONFIG` in `NER_Pipeline_GLiNER.py` and `reset_db.py` with your Supabase keys.

## Scripts
- **`NER_Pipeline_GLiNER.py`**: The main processing engine. Runs locally on your CPU/GPU (No API keys needed). Uses multi-streaming for fast database updates.
- **`reset_db.py`**: Wipes the masking results in the `NLP Gliner` table and resets status to `not_started`.
- **`Discovery_Schema.py`**: Helper tool to list tables and columns in your project.

## Run
```bash
python NER_Pipeline_GLiNER.py
```
The script will loop until all records are processed.
