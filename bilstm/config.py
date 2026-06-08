"""
Central configuration for the BiLSTM Supply Chain Relation Extraction project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE, override=True)
    _ENV_SOURCE = str(_ENV_FILE)
else:
    load_dotenv(override=True)
    _ENV_SOURCE = "no .env file (using process environment only)"

# ---------------------------------------------------------------------------
# Device (MPS on Apple Silicon, else CPU)
# ---------------------------------------------------------------------------
def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

# ---------------------------------------------------------------------------
# Entity markers (must match data in Supabase)
# ---------------------------------------------------------------------------
ENTITY_FROM = "[__NE_FROM__]"
ENTITY_TO = "[__NE_TO__]"
SPECIAL_TOKENS = [ENTITY_FROM, ENTITY_TO]

# ---------------------------------------------------------------------------
# Relation labels (5class | 4class merges supplies 1+2)
# ---------------------------------------------------------------------------
from labels import get_label_mode, get_label_names, get_num_classes

LABEL_MODE = get_label_mode()
NUM_CLASSES = get_num_classes()
LABEL_NAMES = get_label_names()

# ---------------------------------------------------------------------------
# Model & training hyper-parameters
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("TOKENIZER_MODEL", "bert-base-uncased")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "128"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "128"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "256"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", "2"))
DROPOUT = float(os.getenv("DROPOUT", "0.3"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
EPOCHS = int(os.getenv("EPOCHS", "30"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "5"))
LR_SCHEDULER_FACTOR = float(os.getenv("LR_SCHEDULER_FACTOR", "0.5"))
LR_SCHEDULER_PATIENCE = int(os.getenv("LR_SCHEDULER_PATIENCE", "2"))
GRAD_CLIP_NORM = float(os.getenv("GRAD_CLIP_NORM", "1.0"))

USE_CLASS_WEIGHTS = os.getenv("USE_CLASS_WEIGHTS", "true").lower() in ("1", "true", "yes")
USE_WEIGHTED_SAMPLER = os.getenv("USE_WEIGHTED_SAMPLER", "false").lower() in ("1", "true", "yes")
CLASS_WEIGHT_POWER = float(os.getenv("CLASS_WEIGHT_POWER", "0.5"))
MAX_CLASS_WEIGHT = float(os.getenv("MAX_CLASS_WEIGHT", "3.0"))

USE_FOCAL_LOSS = os.getenv("USE_FOCAL_LOSS", "false").lower() in ("1", "true", "yes")
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", "2.0"))
EARLY_STOP_METRIC = os.getenv("EARLY_STOP_METRIC", "macro_f1").strip().lower()
TOP_K_CHECKPOINTS = int(os.getenv("TOP_K_CHECKPOINTS", "3"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "bilstm")

MIN_EPOCHS = int(os.getenv("MIN_EPOCHS", "10"))
# Set NUM_WORKERS to 0 by default to prevent multiprocessing crashes on macOS
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))
PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", "2"))

VAL_RATIO = float(os.getenv("VAL_RATIO", "0.15"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "bilstm/bilstm_relation.pt")
TOKENIZER_SAVE_DIR = os.getenv("TOKENIZER_SAVE_DIR", "bilstm/tokenizer")

# Defaulted to Supabase for the extraction pipeline
DATA_SOURCE = os.getenv("DATA_SOURCE", "supabase").strip().lower()
CSV_PATH = os.getenv("CSV_PATH", str(_PROJECT_ROOT / "final_merged_data.csv")).strip()
CSV_SENTENCE_COL = os.getenv("CSV_SENTENCE_COL", "masked_sentence").strip()
CSV_LABEL_COL = os.getenv("CSV_LABEL_COL", "relation_id").strip()

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip()

# Pointed exactly to your database schema
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "scraped_sentences").strip()
SUPABASE_SENTENCE_COL = os.getenv("SUPABASE_SENTENCE_COL", "masked_sentence").strip()
SUPABASE_LABEL_COL = os.getenv("SUPABASE_LABEL_COL", "relation_id").strip()
SUPABASE_RAW_COL = os.getenv("SUPABASE_RAW_COL", "raw_sentence").strip()
SUPABASE_ENTITY_FROM_COL = os.getenv("SUPABASE_ENTITY_FROM_COL", "entity_from").strip()
SUPABASE_ENTITY_TO_COL = os.getenv("SUPABASE_ENTITY_TO_COL", "entity_to").strip()

BUILD_MASKED_FROM_ENTITIES = os.getenv("BUILD_MASKED_FROM_ENTITIES", "true").lower() in ("1", "true", "yes")

def _mask_secret(value: str, visible: int = 4) -> str:
    if not value:
        return "(empty)"
    if len(value) <= visible * 2:
        return "***"
    return f"{value[:visible]}...{value[-visible:]}"

def _log_config() -> None:
    import multiprocessing
    if multiprocessing.current_process().name != "MainProcess":
        return
    print(f"[config] Env source: {_ENV_SOURCE}")
    print(f"[config] DATA_SOURCE={DATA_SOURCE}")
    print(f"[config] LABEL_MODE={LABEL_MODE} ({NUM_CLASSES} classes)")
    if DATA_SOURCE == "csv":
        print(f"[config] CSV_PATH={CSV_PATH}")
    else:
        print(f"[config] SUPABASE_URL={SUPABASE_URL or '(empty)'}")
        print(f"[config] SUPABASE_TABLE={SUPABASE_TABLE}")
        print(f"[config] SUPABASE_KEY={_mask_secret(SUPABASE_KEY)}")

_log_config()