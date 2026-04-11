"""
ocr.py — Tesseract OCR Wrapper Module (Enhanced)

This module provides an optimized wrapper around pytesseract with:
    - Best engine mode (LSTM neural net via --oem 3)
    - Multi-pass OCR: tries multiple page segmentation modes (PSM)
      and picks the result with the highest average confidence
    - Word-level confidence scoring for frontend highlighting

Usage:
    from app.ocr import extract_text, extract_text_with_confidence
    raw_text = extract_text(preprocessed_image_array)
    detailed = extract_text_with_confidence(preprocessed_image_array)
"""

import pytesseract
from PIL import Image
import numpy as np


# ============================================
# Tesseract Configuration
# ============================================
# --oem 3: Use the LSTM neural net OCR engine (best accuracy)
# PSM modes tried in multi-pass:
#   --psm 6: Assume a single uniform block of text
#   --psm 4: Assume a single column of text
#   --psm 3: Fully automatic page segmentation
TESSERACT_OEM = 3
PSM_MODES = [6, 4, 3]


def _build_config(psm: int = 6) -> str:
    """
    Build a Tesseract config string with optimal settings.

    Args:
        psm (int): Page segmentation mode.

    Returns:
        str: Tesseract config string.
    """
    return f"--oem {TESSERACT_OEM} --psm {psm}"


def _run_tesseract_safe(pil_image: Image.Image, config: str) -> str:
    """
    Run pytesseract.image_to_string with error handling.

    Args:
        pil_image (PIL.Image): Input image.
        config (str): Tesseract config string.

    Returns:
        str: Extracted text, or empty string on failure.
    """
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
    """
    Run pytesseract.image_to_data with error handling.

    Args:
        pil_image (PIL.Image): Input image.
        config (str): Tesseract config string.

    Returns:
        dict: OCR data dictionary with text, conf, etc.
    """
    try:
        return pytesseract.image_to_data(
            pil_image,
            config=config,
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


def _parse_confidence_data(ocr_data: dict) -> tuple:
    """
    Extract word-level confidence data from Tesseract output.

    Args:
        ocr_data (dict): Raw output from pytesseract.image_to_data().

    Returns:
        tuple: (words_list, average_confidence)
            - words_list: list of {"text": str, "confidence": int}
            - average_confidence: float (0-100)
    """
    words = []
    for i in range(len(ocr_data.get("text", []))):
        word_text = ocr_data["text"][i].strip()
        confidence = int(ocr_data["conf"][i])

        # Skip empty entries and invalid confidence (-1)
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


def extract_text(image: np.ndarray) -> str:
    """
    Run Tesseract OCR on a preprocessed image using multi-pass
    strategy and return the best extracted text.

    Multi-pass strategy: tries multiple PSM modes (6, 4, 3) and
    picks the result with the highest average word confidence.

    Args:
        image (np.ndarray): A preprocessed image as a NumPy array.

    Returns:
        str: The best raw text string extracted from the image.

    Raises:
        TypeError: If the input is not a valid NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected a NumPy array, got {type(image).__name__}."
        )

    pil_image = Image.fromarray(image)

    best_text = ""
    best_confidence = -1.0

    # Try each PSM mode and pick the best result
    for psm in PSM_MODES:
        config = _build_config(psm)

        # Get text and confidence data
        ocr_data = _run_tesseract_data_safe(pil_image, config)
        _, avg_conf = _parse_confidence_data(ocr_data)
        text = _run_tesseract_safe(pil_image, config)

        # Keep the result with the highest average confidence
        if avg_conf > best_confidence and text:
            best_confidence = avg_conf
            best_text = text

    return best_text


def extract_text_with_confidence(image: np.ndarray) -> dict:
    """
    Run Tesseract OCR with multi-pass strategy and return the extracted
    text along with word-level confidence scores.

    Tries PSM modes 6, 4, and 3, picks the one with the highest
    average confidence, and returns full word-level detail.

    Args:
        image (np.ndarray): A preprocessed image as a NumPy array.

    Returns:
        dict: A dictionary containing:
            - "raw_text" (str): The full extracted text string
            - "words" (list[dict]): List of word objects with confidence
            - "average_confidence" (float): Average confidence (0-100)
            - "psm_used" (int): Which PSM mode produced the best result

    Raises:
        TypeError: If the input is not a valid NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected a NumPy array, got {type(image).__name__}."
        )

    pil_image = Image.fromarray(image)

    best_text = ""
    best_words = []
    best_confidence = -1.0
    best_psm = PSM_MODES[0]

    # Try each PSM mode and keep the best
    for psm in PSM_MODES:
        config = _build_config(psm)

        # Get detailed confidence data for this PSM mode
        ocr_data = _run_tesseract_data_safe(pil_image, config)
        words, avg_conf = _parse_confidence_data(ocr_data)
        text = _run_tesseract_safe(pil_image, config)

        # Pick the mode with the highest average confidence
        if avg_conf > best_confidence and text:
            best_confidence = avg_conf
            best_text = text
            best_words = words
            best_psm = psm

    return {
        "raw_text": best_text,
        "words": best_words,
        "average_confidence": best_confidence if best_confidence >= 0 else 0.0,
        "psm_used": best_psm,
    }
