"""
backend/sentiment_model.py
────────────────────────────────────────────────────────────────
Inference wrapper for the fine-tuned SupplyChain FinBERT model.

Replaces the TextBlob lexicon-based sentiment in real_data_sources.py
with a genuine DL-based sentiment model.

Key design decisions:
  - Model loaded ONCE as a module-level singleton (not on every call)
  - st.cache_resource compatible — safe for Streamlit multi-user sessions
  - Graceful fallback to TextBlob if model unavailable (HF offline, etc.)
  - Returns a polarity score in [-1.0, +1.0] to preserve compatibility
    with the existing feature pipeline (region_news_sentiment range)

Score mapping from 3-class probabilities to [-1, +1]:
    polarity = P(positive) - P(negative)

    Examples:
      P(neg=0.80, neu=0.15, pos=0.05) → -0.75   (strongly negative)
      P(neg=0.10, neu=0.80, pos=0.10) →  0.00   (neutral)
      P(neg=0.05, neu=0.10, pos=0.85) → +0.80   (strongly positive)

Usage:
    from backend.sentiment_model import get_sentiment_score
    score = get_sentiment_score("Conflict escalates near key port")
    # → -0.72
"""

import os
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Model configuration ───────────────────────────────────────────
# Set HF_REPO env var to your HuggingFace model repo
# e.g. "arunabhachanda/supplychain-finbert"
HF_REPO          = os.environ.get("HF_FINBERT_REPO", "arunabhachanda/supplychain-finbert")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "supplychain_finbert")
MAX_LEN          = 128

# Module-level singleton — loaded once, reused across all calls
_pipeline        = None
_model_source    = None   # "local" | "hub" | "textblob_fallback"


def _load_pipeline():
    """
    Load the sentiment pipeline. Priority:
      1. Local fine-tuned model (data/supplychain_finbert/)
      2. HuggingFace Hub model
      3. Fallback: TextBlob (original implementation)

    Returns the pipeline object or None if falling back to TextBlob.
    """
    global _pipeline, _model_source

    if _pipeline is not None:
        return _pipeline

    # ── Try local model first (fastest, no network) ───────────────
    if os.path.isdir(LOCAL_MODEL_PATH):
        try:
            from transformers import pipeline
            logger.info(f"Loading sentiment model from local path: {LOCAL_MODEL_PATH}")
            _pipeline = pipeline(
                "text-classification",
                model=LOCAL_MODEL_PATH,
                tokenizer=LOCAL_MODEL_PATH,
                return_all_scores=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            _model_source = "local"
            logger.info("✅ FinBERT sentiment model loaded from local path")
            return _pipeline
        except Exception as e:
            logger.warning(f"Local model load failed: {e}")

    # ── Try HuggingFace Hub ───────────────────────────────────────
    try:
        from transformers import pipeline
        logger.info(f"Loading sentiment model from HuggingFace Hub: {HF_REPO}")
        _pipeline = pipeline(
            "text-classification",
            model=HF_REPO,
            return_all_scores=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        _model_source = "hub"
        logger.info(f"✅ FinBERT sentiment model loaded from HuggingFace Hub: {HF_REPO}")
        return _pipeline

    except Exception as e:
        logger.warning(
            f"HuggingFace Hub model unavailable ({e}). "
            f"Falling back to TextBlob sentiment analysis."
        )
        _model_source = "textblob_fallback"
        return None


def _textblob_fallback(text: str) -> float:
    """Original TextBlob-based sentiment as a fallback."""
    try:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def get_sentiment_score(text: str) -> float:
    """
    Get sentiment polarity score for a single text string.

    Args:
        text: News headline or article text

    Returns:
        float in [-1.0, +1.0]
          -1.0 = maximally negative (high risk signal)
           0.0 = neutral
          +1.0 = maximally positive (low risk signal)
    """
    pipe = _load_pipeline()

    if pipe is None:
        # TextBlob fallback
        return _textblob_fallback(text)

    try:
        # Pipeline returns: [{"label": "negative", "score": 0.8}, ...]
        results = pipe(text)[0]

        # Build probability dict regardless of label order
        probs = {r["label"].lower(): r["score"] for r in results}

        p_neg = probs.get("negative", 0.0)
        p_pos = probs.get("positive", 0.0)

        # Polarity: ranges from -1 (all negative) to +1 (all positive)
        polarity = p_pos - p_neg

        return float(polarity)

    except Exception as e:
        logger.warning(f"FinBERT inference failed: {e}. Using TextBlob fallback.")
        return _textblob_fallback(text)


def get_sentiment_scores_batch(texts: List[str], batch_size: int = 32) -> List[float]:
    """
    Batch inference for multiple texts — significantly faster than
    calling get_sentiment_score() in a loop.

    Args:
        texts:      List of headline strings
        batch_size: Number of texts per forward pass

    Returns:
        List of polarity scores in [-1.0, +1.0]
    """
    pipe = _load_pipeline()

    if pipe is None:
        return [_textblob_fallback(t) for t in texts]

    try:
        scores = []
        for i in range(0, len(texts), batch_size):
            batch   = texts[i: i + batch_size]
            results = pipe(batch)  # returns list of lists when input is a list
            for result in results:
                probs  = {r["label"].lower(): r["score"] for r in result}
                p_neg  = probs.get("negative", 0.0)
                p_pos  = probs.get("positive", 0.0)
                scores.append(float(p_pos - p_neg))
        return scores

    except Exception as e:
        logger.warning(f"Batch FinBERT inference failed: {e}. Using TextBlob fallback.")
        return [_textblob_fallback(t) for t in texts]


def get_model_info() -> dict:
    """
    Returns metadata about the currently loaded sentiment model.
    Used by the Data Provenance panel in the Streamlit UI.
    """
    _load_pipeline()  # Ensure model is loaded
    return {
        "model":       "SupplyChain FinBERT (fine-tuned)" if _model_source != "textblob_fallback" else "TextBlob (fallback)",
        "base_model":  "ProsusAI/finbert" if _model_source != "textblob_fallback" else "N/A",
        "source":      _model_source or "unknown",
        "hf_repo":     HF_REPO if _model_source == "hub" else ("local" if _model_source == "local" else "N/A"),
        "labels":      ["negative", "neutral", "positive"] if _model_source != "textblob_fallback" else ["N/A"],
        "score_range": "[-1.0, +1.0]",
        "method":      "P(positive) - P(negative)" if _model_source != "textblob_fallback" else "TextBlob lexicon polarity",
    }
