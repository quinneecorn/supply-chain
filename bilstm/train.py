"""
Training loop and orchestration entry point.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    EARLY_STOP_METRIC,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    FOCAL_GAMMA,
    GRAD_CLIP_NORM,
    LABEL_MODE,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MIN_EPOCHS,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    TOKENIZER_SAVE_DIR,
    TOP_K_CHECKPOINTS,
    USE_CLASS_WEIGHTS,
    USE_FOCAL_LOSS,
    WEIGHT_DECAY,
)
from data_loader import (
    build_tokenizer,
    compute_class_weights,
    create_dataloaders,
    load_training_samples,
)
from losses import FocalLoss
from model import BiLSTMClassifier


def train_one_epoch(
    model: BiLSTMClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_epoch(
    model: BiLSTMClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Return (unweighted val loss, macro F1)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / max(num_batches, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return val_loss, macro_f1


def save_checkpoint(
    model: BiLSTMClassifier,
    tokenizer_vocab_size: int,
    path: str = MODEL_SAVE_PATH,
    epoch: int = 0,
    macro_f1: float = 0.0,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer_vocab_size,
            "num_classes": model.num_classes,
            "label_mode": LABEL_MODE,
            "config": {
                "embed_dim": model.embed_dim,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
            },
            "epoch": epoch,
            "macro_f1": macro_f1,
        },
        path,
    )
    print(f"[train] Checkpoint saved to {path}")


def _update_top_k_checkpoints(
    top_checkpoints: list[tuple[float, int, str]],
    macro_f1: float,
    epoch: int,
    model: BiLSTMClassifier,
    vocab_size: int,
) -> list[tuple[float, int, str]]:
    """Keep top-K checkpoints by macro F1 on disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(
        CHECKPOINT_DIR, f"bilstm_ep{epoch:02d}_f1_{macro_f1:.4f}.pt"
    )
    save_checkpoint(model, vocab_size, path, epoch=epoch, macro_f1=macro_f1)
    top_checkpoints.append((macro_f1, epoch, path))
    top_checkpoints.sort(key=lambda x: x[0], reverse=True)

    while len(top_checkpoints) > TOP_K_CHECKPOINTS:
        _, _, old_path = top_checkpoints.pop()
        if old_path != MODEL_SAVE_PATH and os.path.isfile(old_path):
            os.remove(old_path)
            print(f"[train] Removed lower checkpoint {old_path}")

    return top_checkpoints


def _build_train_criterion(
    train_samples: list[dict],
    device: str,
) -> nn.Module:
    loss_weights = None
    if USE_CLASS_WEIGHTS:
        loss_weights = compute_class_weights(train_samples).to(device)

    if USE_FOCAL_LOSS:
        return FocalLoss(alpha=loss_weights, gamma=FOCAL_GAMMA)
    return nn.CrossEntropyLoss(weight=loss_weights)


def main() -> None:
    print(f"[train] Using device: {DEVICE}")

    samples = load_training_samples()
    tokenizer = build_tokenizer(save_dir=TOKENIZER_SAVE_DIR)
    train_loader, val_loader, train_samples = create_dataloaders(
        samples, tokenizer, batch_size=BATCH_SIZE
    )

    vocab_size = len(tokenizer)
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        num_classes=NUM_CLASSES,
        padding_idx=tokenizer.pad_token_id or 0,
    ).to(DEVICE)

    criterion_train = _build_train_criterion(train_samples, DEVICE)
    criterion_val = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    use_f1_stop = EARLY_STOP_METRIC == "macro_f1"
    print(
        f"[train] Imbalanced mode: focal={USE_FOCAL_LOSS}, class_weights={USE_CLASS_WEIGHTS}, "
        f"early_stop_metric={EARLY_STOP_METRIC}"
    )
    print(
        f"[train] epochs={EPOCHS}, min_epochs={MIN_EPOCHS}, "
        f"patience={EARLY_STOP_PATIENCE}, batch={BATCH_SIZE}"
    )

    best_score = float("-inf") if use_f1_stop else float("inf")
    epochs_without_improve = 0
    top_checkpoints: list[tuple[float, int, str]] = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, criterion_train, optimizer, DEVICE
        )
        val_loss, macro_f1 = validate_epoch(
            model, val_loader, criterion_val, DEVICE
        )
        scheduler.step(val_loss)
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        improved = (
            macro_f1 > best_score if use_f1_stop else val_loss < best_score
        )
        score_label = "macro_f1" if use_f1_stop else "val_loss"

        print(
            f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | macro_f1={macro_f1:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        if improved:
            best_score = macro_f1 if use_f1_stop else val_loss
            epochs_without_improve = 0
            save_checkpoint(
                model,
                vocab_size,
                MODEL_SAVE_PATH,
                epoch=epoch,
                macro_f1=macro_f1,
            )
            if use_f1_stop:
                top_checkpoints = _update_top_k_checkpoints(
                    top_checkpoints, macro_f1, epoch, model, vocab_size
                )
            print(f"[train] New best {score_label}={best_score:.4f}")
        else:
            epochs_without_improve += 1
            if epoch >= MIN_EPOCHS and epochs_without_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"[train] Early stopping at epoch {epoch} "
                    f"(best {score_label}={best_score:.4f})."
                )
                break

    print(f"[train] Done. Best {score_label}={best_score:.4f}")
    if top_checkpoints:
        print("[train] Top checkpoints by macro F1:")
        for f1, ep, pth in top_checkpoints:
            print(f"  epoch {ep}: macro_f1={f1:.4f} -> {pth}")


if __name__ == "__main__":
    main()
