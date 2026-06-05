"""
Evaluation metrics and inference for relation extraction.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from config import (
    DEVICE,
    LABEL_MODE,
    LABEL_NAMES,
    MAX_SEQ_LEN,
    MODEL_SAVE_PATH,
    TOKENIZER_SAVE_DIR,
)
from labels import five_class_name, get_label_mode, get_num_classes, to_five_class_id
from data_loader import RelationDataset, load_tokenizer_from_checkpoint
from model import BiLSTMClassifier


def load_model(
    checkpoint_path: str = MODEL_SAVE_PATH,
    device: str = DEVICE,
) -> tuple[BiLSTMClassifier, dict[str, Any]]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    num_classes = checkpoint.get("num_classes", get_num_classes())
    model = BiLSTMClassifier(
        vocab_size=vocab_size, num_classes=num_classes
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(
    model: BiLSTMClassifier,
    loader: DataLoader,
    device: str = DEVICE,
) -> dict[str, float]:
    """
    Compute macro-averaged Precision, Recall, and F1 on a DataLoader.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    metrics = {
        "precision": precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    print(
        f"Macro Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )
    target_names = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    print("\n" + classification_report(
        all_labels, all_preds, target_names=target_names, digits=4
    ))
    if get_label_mode() == "4class":
        print("(4-class mode: supplies direction resolved at inference via labels.py)")
    return metrics


@torch.no_grad()
def predict_relation(
    sentence: str,
    model: BiLSTMClassifier | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    device: str = DEVICE,
    checkpoint_path: str = MODEL_SAVE_PATH,
) -> dict[str, Any]:
    """
    Run inference on a single masked sentence.

    Returns dict with predicted class id, label name, and class probabilities.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer_from_checkpoint(TOKENIZER_SAVE_DIR)
    if model is None:
        model, _ = load_model(checkpoint_path, device)

    encoding = tokenizer(
        sentence,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    pred_id = int(probs.argmax().item())
    five_class_id = to_five_class_id(pred_id, sentence)

    return {
        "class_id": pred_id,
        "label": LABEL_NAMES[pred_id],
        "five_class_id": five_class_id,
        "five_class_label": five_class_name(five_class_id),
        "label_mode": get_label_mode(),
        "probabilities": {
            LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))
        },
    }


def main() -> None:
    """Evaluate best checkpoint on validation split."""
    import argparse

    from data_loader import create_dataloaders, load_training_samples

    parser = argparse.ArgumentParser(description="Evaluate BiLSTM relation model")
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV path (default: CSV_PATH from .env)",
    )
    parser.add_argument(
        "--checkpoint",
        default=MODEL_SAVE_PATH,
        help="Checkpoint .pt file",
    )
    args = parser.parse_args()

    if args.csv:
        print(f"[evaluate] Using CSV: {args.csv}")
    samples = load_training_samples(csv_path=args.csv)
    tokenizer = load_tokenizer_from_checkpoint()
    _, val_loader, _ = create_dataloaders(samples, tokenizer)

    model, _ = load_model(args.checkpoint)
    evaluate(model, val_loader)


if __name__ == "__main__":
    main()
