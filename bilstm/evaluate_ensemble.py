"""
Evaluate top-K checkpoints with majority vote on validation set.
"""

from __future__ import annotations

import os

# Avoid multiprocessing issues in quick eval scripts
os.environ.setdefault("NUM_WORKERS", "0")

import glob
from collections import Counter

import torch
from sklearn.metrics import classification_report, f1_score

from config import CHECKPOINT_DIR, DEVICE, LABEL_NAMES
from data_loader import create_dataloaders, load_training_samples
from evaluate import load_model
from model import BiLSTMClassifier
from data_loader import load_tokenizer_from_checkpoint


def _find_top_checkpoints(k: int = 3) -> list[str]:
    pattern = os.path.join(CHECKPOINT_DIR, "bilstm_ep*_f1_*.pt")
    files = glob.glob(pattern)
    # fallback old naming bilstm_ep10_f10.5930.pt
    files.extend(glob.glob(os.path.join(CHECKPOINT_DIR, "bilstm_ep*_f1*.pt")))
    files = sorted(set(files), reverse=True)

    scored: list[tuple[float, str]] = []
    for path in files:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        f1 = float(ckpt.get("macro_f1", 0))
        scored.append((f1, path))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


@torch.no_grad()
def main() -> None:
    paths = _find_top_checkpoints(3)
    if not paths:
        print("No top checkpoints found. Run train.py first.")
        return

    print("Ensemble checkpoints:")
    for p in paths:
        print(f"  {p}")

    samples = load_training_samples()
    tokenizer = load_tokenizer_from_checkpoint()
    _, val_loader, _ = create_dataloaders(samples, tokenizer)

    models: list[BiLSTMClassifier] = []
    for path in paths:
        model, ckpt = load_model(path, DEVICE)
        models.append(model)

    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"]

        votes: list[list[int]] = []
        for model in models:
            logits = model(input_ids, attention_mask)
            votes.append(logits.argmax(dim=-1).cpu().tolist())

        batch_size = labels.size(0)
        for i in range(batch_size):
            ballot = Counter(v[i] for v in votes)
            pred = ballot.most_common(1)[0][0]
            all_preds.append(pred)
            all_labels.append(int(labels[i].item()))

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\nEnsemble ({len(models)} models) macro F1: {macro_f1:.4f}")
    names = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    print(classification_report(all_labels, all_preds, target_names=names, digits=4))


if __name__ == "__main__":
    main()
