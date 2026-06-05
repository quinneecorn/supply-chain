"""
Label schema: 5-class (original) or 4-class (merge supplies 1+2).
"""

from __future__ import annotations

import os

ENTITY_FROM = "[__NE_FROM__]"
ENTITY_TO = "[__NE_TO__]"

LABEL_NAMES_5: dict[int, str] = {
    0: "Reject / No relation",
    1: "B supplies A",
    2: "A supplies B",
    3: "Non directed / Partnership",
    4: "Ownership",
}

LABEL_NAMES_4: dict[int, str] = {
    0: "Reject / No relation",
    1: "Supplies (merged 1+2)",
    2: "Non directed / Partnership",
    3: "Ownership",
}


def get_label_mode() -> str:
    return os.getenv("LABEL_MODE", "5class").strip().lower()


def get_num_classes() -> int:
    return 4 if get_label_mode() == "4class" else 5


def get_label_names() -> dict[int, str]:
    return LABEL_NAMES_4 if get_label_mode() == "4class" else LABEL_NAMES_5


def normalize_label(raw_label: int) -> int | None:
    """Map original relation_id (0-4) to training class index."""
    if raw_label not in range(5):
        return None
    if get_label_mode() == "5class":
        return raw_label
    if raw_label == 0:
        return 0
    if raw_label in (1, 2):
        return 1
    if raw_label == 3:
        return 2
    if raw_label == 4:
        return 3
    return None


def infer_supply_direction(sentence: str) -> int:
    """
    Split merged Supplies back to 5-class: 1 = B supplies A, 2 = A supplies B.
    Heuristic: FROM marker before TO -> class 1, else class 2.
    """
    i_from = sentence.find(ENTITY_FROM)
    i_to = sentence.find(ENTITY_TO)
    if i_from < 0 or i_to < 0:
        return 1
    return 1 if i_from < i_to else 2


def to_five_class_id(model_class: int, sentence: str) -> int:
    """Convert model output to original 5-class id for reporting/API."""
    if get_label_mode() == "5class":
        return model_class
    if model_class == 0:
        return 0
    if model_class == 1:
        return infer_supply_direction(sentence)
    if model_class == 2:
        return 3
    if model_class == 3:
        return 4
    return 0


def five_class_name(class_id: int) -> str:
    return LABEL_NAMES_5.get(class_id, "Unknown")
