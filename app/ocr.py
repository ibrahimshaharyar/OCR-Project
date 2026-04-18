"""
ocr.py — Tesseract OCR Wrapper Module (Dual-Pipeline)

This module runs Tesseract OCR using a smart comparison strategy:
    1. Takes TWO preprocessed images (simple + enhanced pipelines)
    2. Runs OCR on both with multiple PSM modes
    3. Scores each result by text quality (not just confidence)
    4. Returns the result with the best actual text extraction

Quality scoring uses:
    - Number of real words detected (more = better)
    - Average confidence (higher = better)
    - Total text length (longer = more content captured)

Usage:
    from app.ocr import extract_text, extract_text_with_confidence
    raw_text = extract_text(preprocessed_image)
    detailed = extract_text_with_confidence(preprocessed_image)
    # Or with dual pipeline:
    from app.ocr import extract_best_from_dual
    result = extract_best_from_dual(simple_img, enhanced_img)
"""

import pytesseract
from PIL import Image
import numpy as np
import re


# ============================================
# Tesseract Configuration
# ============================================
TESSERACT_OEM = 3  # LSTM neural net engine (best accuracy)
PSM_MODES = [6, 4, 3]  # Modes to try: uniform block, single column, auto


def _build_config(psm: int = 6) -> str:
    """Build Tesseract config string."""
    return f"--oem {TESSERACT_OEM} --psm {psm}"


# ============================================
# Safe Tesseract Wrappers
# ============================================

def _run_tesseract_safe(pil_image: Image.Image, config: str) -> str:
    """Run image_to_string with error handling."""
    try:
        return pytesseract.image_to_string(pil_image, config=config).strip()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR engine is not installed or not found in PATH. "
            "Install it via: brew install tesseract (Mac), "
            "sudo apt install tesseract-ocr (Linux), "
            "or download from https://github.com/UB-Mannheim/tesseract/wiki (Windows)."
        )
    except Exception:
        return ""


def _run_tesseract_data_safe(pil_image: Image.Image, config: str) -> dict:
    """Run image_to_data with error handling."""
    try:
        return pytesseract.image_to_data(
            pil_image, config=config,
            output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR engine is not installed or not found in PATH. "
            "Install it via: brew install tesseract (Mac), "
            "sudo apt install tesseract-ocr (Linux), "
            "or download from https://github.com/UB-Mannheim/tesseract/wiki (Windows)."
        )
    except Exception:
        return {"text": [], "conf": []}


# ============================================
# Quality Scoring
# ============================================

def _parse_confidence_data(ocr_data: dict) -> tuple:
    """
    Extract word-level confidence data from Tesseract output.

    Returns:
        tuple: (words_list, average_confidence)
    """
    words = []
    for i in range(len(ocr_data.get("text", []))):
        word_text = ocr_data["text"][i].strip()
        confidence = int(ocr_data["conf"][i])

        if word_text and confidence >= 0:
            words.append({
                "text": word_text,
                "confidence": confidence,
            })

    if words:
        avg = round(sum(w["confidence"] for w in words) / len(words), 1)
    else:
        avg = 0.0

    return words, avg


def _score_result(text: str, words: list, avg_confidence: float) -> float:
    """
    Score an OCR result by actual text quality, not just confidence.

    The scoring formula balances three factors:
        1. Real word count (40% weight) — more recognizable words = better
        2. Average confidence (30% weight) — but not dominant
        3. Text length (30% weight) — more content captured = better

    A "real word" is defined as 2+ alphanumeric characters. This filters
    out OCR noise like single random characters.

    Args:
        text (str): Raw OCR text.
        words (list): Word-confidence pairs.
        avg_confidence (float): Average confidence score.

    Returns:
        float: Quality score (higher = better).
    """
    if not text:
        return 0.0

    # Count "real words" — 2+ alphanumeric chars (filters noise)
    real_words = re.findall(r'[a-zA-Z0-9]{2,}', text)
    word_count = len(real_words)

    # Text length (chars) — more content = more was captured
    text_length = len(text.strip())

    # Weighted score
    # - word_count is the strongest signal (captures text completeness)
    # - confidence matters but shouldn't dominate (can be misleadingly high)
    # - text_length rewards capturing more of the receipt
    score = (
        (word_count * 4.0) +          # 40% — real word count
        (avg_confidence * 0.3) +      # 30% — confidence
        (text_length * 0.03)           # 30% — text length
    )

    return round(score, 2)


# ============================================
# Single-Image OCR (backward compatible)
# ============================================

def extract_text(image: np.ndarray) -> str:
    """
    Run Tesseract OCR on a single preprocessed image.
    Tries multiple PSM modes and picks the best result.

    Args:
        image (np.ndarray): Preprocessed image.

    Returns:
        str: Best extracted text.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected a NumPy array, got {type(image).__name__}.")

    pil_image = Image.fromarray(image)
    best_text = ""
    best_score = -1.0

    for psm in PSM_MODES:
        config = _build_config(psm)
        text = _run_tesseract_safe(pil_image, config)
        ocr_data = _run_tesseract_data_safe(pil_image, config)
        words, avg_conf = _parse_confidence_data(ocr_data)
        score = _score_result(text, words, avg_conf)

        if score > best_score and text:
            best_score = score
            best_text = text

    return best_text


def extract_text_with_confidence(image: np.ndarray) -> dict:
    """
    Run Tesseract OCR on a single preprocessed image and return
    text with word-level confidence scores.

    Args:
        image (np.ndarray): Preprocessed image.

    Returns:
        dict: {raw_text, words, average_confidence, psm_used}
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected a NumPy array, got {type(image).__name__}.")

    pil_image = Image.fromarray(image)
    best_text = ""
    best_words = []
    best_confidence = 0.0
    best_score = -1.0
    best_psm = PSM_MODES[0]

    for psm in PSM_MODES:
        config = _build_config(psm)
        text = _run_tesseract_safe(pil_image, config)
        ocr_data = _run_tesseract_data_safe(pil_image, config)
        words, avg_conf = _parse_confidence_data(ocr_data)
        score = _score_result(text, words, avg_conf)

        if score > best_score and text:
            best_score = score
            best_text = text
            best_words = words
            best_confidence = avg_conf
            best_psm = psm

    return {
        "raw_text": best_text,
        "words": best_words,
        "average_confidence": best_confidence,
        "psm_used": best_psm,
    }


# ============================================
# Dual-Pipeline OCR (the main improvement)
# ============================================

def extract_best_from_dual(simple_img: np.ndarray, enhanced_img: np.ndarray) -> dict:
    """
    Run OCR on BOTH simple and enhanced preprocessed images,
    trying multiple PSM modes on each, and return the single
    best result based on actual text quality scoring.

    This is the core dual-pipeline strategy:
        - 2 images × 3 PSM modes = 6 OCR attempts
        - Score each by word count + confidence + text length
        - Return the winner

    Args:
        simple_img (np.ndarray): Image from simple pipeline (Otsu).
        enhanced_img (np.ndarray): Image from enhanced pipeline (adaptive).

    Returns:
        dict: Best result with:
            - raw_text (str)
            - words (list[dict])
            - average_confidence (float)
            - psm_used (int)
            - pipeline_used (str): "simple" or "enhanced"
    """
    candidates = []

    # Run all combinations
    for label, img in [("simple", simple_img), ("enhanced", enhanced_img)]:
        pil_image = Image.fromarray(img)

        for psm in PSM_MODES:
            config = _build_config(psm)
            text = _run_tesseract_safe(pil_image, config)
            ocr_data = _run_tesseract_data_safe(pil_image, config)
            words, avg_conf = _parse_confidence_data(ocr_data)
            score = _score_result(text, words, avg_conf)

            if text:
                candidates.append({
                    "raw_text": text,
                    "words": words,
                    "average_confidence": avg_conf,
                    "psm_used": psm,
                    "pipeline_used": label,
                    "score": score,
                })

    # Pick the best candidate by score
    if not candidates:
        return {
            "raw_text": "",
            "words": [],
            "average_confidence": 0.0,
            "psm_used": 6,
            "pipeline_used": "none",
        }

    best = max(candidates, key=lambda c: c["score"])

    # Remove internal score from the result
    best.pop("score", None)
    return best
