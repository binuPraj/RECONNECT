"""Named Entity Recognition (NER) Service using fine-tuned DeBERTa model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "deberta-ner"

_pipeline = None


def get_ner_pipeline():
    """Lazily load and return the DeBERTa token classification pipeline."""
    global _pipeline
    if _pipeline is None:
        if not MODEL_DIR.exists():
            LOGGER.warning("[ner] Model directory not found at %s", MODEL_DIR)
            return None
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
            import torch

            LOGGER.info("[ner] Loading DeBERTa NER model from %s", MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
            model.eval()
            device = 0 if torch.cuda.is_available() else -1
            _pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device,
            )
            LOGGER.info("[ner] DeBERTa NER model loaded successfully")
        except Exception:
            LOGGER.exception("[ner] Failed to load DeBERTa NER model from %s", MODEL_DIR)
            return None
    return _pipeline


def extract_entities(text: str) -> list[dict[str, str]]:
    """Extract named entities from raw summary text using the fine-tuned DeBERTa model.

    Args:
        text: The summarized text string.

    Returns:
        A list of distinct entities: [{"type": "PERSON", "text": "Kris"}, ...]
        Supported entity types: PERSON, PLACE, DATE, TIME, EVENT, ITEM, ORGANIZATION.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []

    pipe = get_ner_pipeline()
    if pipe is None:
        LOGGER.warning("[ner] Pipeline unavailable; returning empty entity list")
        return []

    try:
        raw_results: list[dict[str, Any]] = pipe(text)
    except Exception:
        LOGGER.exception("[ner] Inference failed for text: %s", text[:80])
        return []

    entities: list[dict[str, str]] = []
    seen = set()

    for item in raw_results:
        entity_type = str(item.get("entity_group", "")).upper().strip()
        word = str(item.get("word", "")).strip()

        # Clean surrounding punctuation
        word = word.strip(" ,.;:\"'()[]{}")
        if not word or not entity_type or entity_type == "O":
            continue

        pair = (entity_type, word)
        if pair not in seen:
            seen.add(pair)
            entities.append({"type": entity_type, "text": word})

    return entities


def __getattr__(name: str) -> Any:
    """Provide backward compatibility for direct attribute access (e.g. model, tokenizer)."""
    if name in ("model", "tokenizer"):
        pipe = get_ner_pipeline()
        if pipe is not None:
            if name == "model":
                return pipe.model
            if name == "tokenizer":
                return pipe.tokenizer
        return None
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
