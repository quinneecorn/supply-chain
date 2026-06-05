"""
Data fetching from Supabase and PyTorch Dataset / DataLoader utilities.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from transformers import AutoTokenizer, PreTrainedTokenizer

from config import (
    BATCH_SIZE,
    BUILD_MASKED_FROM_ENTITIES,
    CSV_LABEL_COL,
    CSV_PATH,
    CSV_SENTENCE_COL,
    DATA_SOURCE,
    ENTITY_FROM,
    ENTITY_TO,
    MAX_SEQ_LEN,
    MODEL_NAME,
    CLASS_WEIGHT_POWER,
    MAX_CLASS_WEIGHT,
    NUM_CLASSES,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    RANDOM_SEED,
    SPECIAL_TOKENS,
    USE_WEIGHTED_SAMPLER,
    SUPABASE_ENTITY_FROM_COL,
    SUPABASE_ENTITY_TO_COL,
    SUPABASE_KEY,
    SUPABASE_LABEL_COL,
    SUPABASE_RAW_COL,
    SUPABASE_SENTENCE_COL,
    SUPABASE_TABLE,
    SUPABASE_URL,
    TOKENIZER_SAVE_DIR,
    VAL_RATIO,
)
from labels import normalize_label


def _is_valid_sentence(text: Any) -> bool:
    if text is None:
        return False
    s = str(text).strip()
    return s not in ("", "None", "null")


def _build_masked_from_entities(
    raw_sentence: str,
    entity_from: str,
    entity_to: str,
) -> str | None:
    """Replace entity spans in raw text with [__NE_FROM__] / [__NE_TO__]."""
    if not raw_sentence or not entity_from or not entity_to:
        return None
    replacements = [(entity_from, ENTITY_FROM), (entity_to, ENTITY_TO)]
    # Replace longer spans first so nested/overlapping names are handled safely
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)

    masked = raw_sentence
    for span, token in replacements:
        if span not in masked:
            return None
        masked = masked.replace(span, token, 1)

    if ENTITY_FROM in masked and ENTITY_TO in masked:
        return masked
    return None


def _rows_to_samples(
    rows: list[dict[str, Any]],
    sentence_col: str,
    label_col: str,
    source_name: str,
) -> list[dict[str, Any]]:
    """Shared filter: both entity markers + relation_id in 0..4."""
    samples: list[dict[str, Any]] = []
    skipped = 0
    built_from_entities = 0

    for row in rows:
        sentence = row.get(sentence_col)
        if _is_valid_sentence(sentence):
            sentence = str(sentence).strip()
        elif BUILD_MASKED_FROM_ENTITIES:
            raw = row.get(SUPABASE_RAW_COL)
            ent_from = row.get(SUPABASE_ENTITY_FROM_COL)
            ent_to = row.get(SUPABASE_ENTITY_TO_COL)
            if not _is_valid_sentence(raw) or not _is_valid_sentence(ent_from):
                skipped += 1
                continue
            if not _is_valid_sentence(ent_to):
                skipped += 1
                continue
            sentence = _build_masked_from_entities(
                str(raw).strip(), str(ent_from).strip(), str(ent_to).strip()
            )
            if sentence is None:
                skipped += 1
                continue
            built_from_entities += 1
        else:
            skipped += 1
            continue

        if ENTITY_FROM not in sentence or ENTITY_TO not in sentence:
            skipped += 1
            continue

        raw_label = row.get(label_col)
        if raw_label is None or str(raw_label).strip() in ("", "None", "null", "nan"):
            skipped += 1
            continue
        try:
            raw_id = int(float(raw_label))
        except (TypeError, ValueError):
            skipped += 1
            continue
        label = normalize_label(raw_id)
        if label is None:
            skipped += 1
            continue
        samples.append({"sentence": sentence, "label": label, "raw_label": raw_id})

    print(
        f"[data_loader] {source_name}: {len(rows)} rows, "
        f"{len(samples)} trainable (skipped {skipped}"
        f"{f', built {built_from_entities} from entity columns' if built_from_entities else ''})."
    )
    if not samples:
        raise RuntimeError(
            f"No trainable rows from {source_name}. "
            f"Need '{sentence_col}' with '{ENTITY_FROM}' and '{ENTITY_TO}', "
            f"and '{label_col}' in 0..4."
        )
    return samples


def fetch_data_from_csv(path: str | None = None) -> list[dict[str, Any]]:
    """Load labeled sentences from a local CSV export."""
    csv_path = Path(path or CSV_PATH)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    print(f"[data_loader] Loading CSV: {csv_path}")
    return _rows_to_samples(rows, CSV_SENTENCE_COL, CSV_LABEL_COL, str(csv_path))


def fetch_data_from_supabase() -> list[dict[str, Any]]:
    """
    Pull labeled sentences from Supabase (paginated).

    Uses columns from config (default: masked_sentence, relation_id).
    Keeps only rows whose text contains [__NE_FROM__] and [__NE_TO__].
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY are missing. "
            "Copy .env.example to .env in the project root and set both values:\n"
            "  cp .env.example .env"
        )

    from supabase import create_client

    print("[data_loader] Connecting to Supabase with:")
    print(f"  SUPABASE_URL={SUPABASE_URL}")
    print(f"  SUPABASE_TABLE={SUPABASE_TABLE}")
    print(f"  SUPABASE_KEY=*** (set={bool(SUPABASE_KEY)})")

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    page_size = 1000
    offset = 0
    rows: list[dict[str, Any]] = []

    select_cols = f"{SUPABASE_SENTENCE_COL}, {SUPABASE_LABEL_COL}"
    if BUILD_MASKED_FROM_ENTITIES:
        select_cols += (
            f", {SUPABASE_RAW_COL}, {SUPABASE_ENTITY_FROM_COL}, {SUPABASE_ENTITY_TO_COL}"
        )

    while True:
        response = (
            client.table(SUPABASE_TABLE)
            .select(select_cols)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not rows:
        raise RuntimeError(f"No rows returned from table '{SUPABASE_TABLE}'.")

    return _rows_to_samples(
        rows, SUPABASE_SENTENCE_COL, SUPABASE_LABEL_COL, f"table '{SUPABASE_TABLE}'"
    )


def load_training_samples(csv_path: str | None = None) -> list[dict[str, Any]]:
    """Load samples from CSV or Supabase based on DATA_SOURCE in config."""
    if DATA_SOURCE == "csv":
        return fetch_data_from_csv(csv_path)
    if DATA_SOURCE == "supabase":
        return fetch_data_from_supabase()
    raise ValueError(f"Unknown DATA_SOURCE={DATA_SOURCE!r}. Use 'csv' or 'supabase'.")


def build_tokenizer(save_dir: str | None = TOKENIZER_SAVE_DIR) -> PreTrainedTokenizer:
    """
    Load a HuggingFace tokenizer and register entity markers as single tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    print(
        f"[data_loader] Tokenizer '{MODEL_NAME}': added {num_added} special tokens "
        f"({ENTITY_FROM}, {ENTITY_TO}). Vocab size = {len(tokenizer)}."
    )

    if save_dir:
        import os

        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        print(f"[data_loader] Tokenizer saved to {save_dir}")

    return tokenizer


class RelationDataset(Dataset):
    """Tokenized sentence + integer relation label."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = MAX_SEQ_LEN,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["sentence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def compute_class_weights(samples: list[dict[str, Any]]) -> torch.Tensor:
    """
    Smoothed inverse-frequency weights for CrossEntropyLoss.

    Uses power < 1 and a cap so minority classes are boosted without
    collapsing majority-class recall (class 0).
    """
    from collections import Counter

    counts = Counter(s["label"] for s in samples)
    total = sum(counts.values())
    raw: list[float] = []
    for c in range(NUM_CLASSES):
        n = counts.get(c, 0)
        raw.append(total / (NUM_CLASSES * n) if n > 0 else 1.0)

    weights = [w**CLASS_WEIGHT_POWER for w in raw]
    weights = [min(w, MAX_CLASS_WEIGHT) for w in weights]
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]

    tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"[data_loader] Class counts: {dict(sorted(counts.items()))}")
    print(
        f"[data_loader] Class weights (power={CLASS_WEIGHT_POWER}, "
        f"cap={MAX_CLASS_WEIGHT}): {[round(w, 3) for w in weights]}"
    )
    return tensor


def _dataloader_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if NUM_WORKERS > 0:
        kwargs["num_workers"] = NUM_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
        kwargs["persistent_workers"] = True
    return kwargs


def create_dataloaders(
    samples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, list[dict[str, Any]]]:
    """Split samples into train/val and return DataLoaders + train subset samples."""
    dataset = RelationDataset(samples, tokenizer)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_samples = [samples[i] for i in train_ds.indices]
    dl_kw = _dataloader_kwargs()

    if USE_WEIGHTED_SAMPLER:
        class_weights = compute_class_weights(train_samples)
        sample_weights = [class_weights[s["label"]].item() for s in train_samples]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, **dl_kw
        )
        print("[data_loader] Using WeightedRandomSampler (minority oversampling).")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, **dl_kw
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)

    print(
        f"[data_loader] Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, num_workers={NUM_WORKERS}"
    )
    return train_loader, val_loader, train_samples


def load_tokenizer_from_checkpoint(path: str = TOKENIZER_SAVE_DIR) -> PreTrainedTokenizer:
    """Reload tokenizer saved during training (includes special tokens)."""
    return AutoTokenizer.from_pretrained(path)
