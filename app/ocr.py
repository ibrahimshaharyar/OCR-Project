"""
ocr.py — Tesseract OCR Wrapper Module

This module provides a simple wrapper around pytesseract to convert
preprocessed images (NumPy arrays) into raw text strings.

It handles the conversion from OpenCV/NumPy format to PIL Image format
(required by pytesseract) and applies OCR configuration optimized
for receipt text extraction.

Usage:
    from app.ocr import extract_text
    raw_text = extract_text(preprocessed_image_array)
"""

import pytesseract
from PIL import Image
import numpy as np


def extract_text(image: np.ndarray) -> str:
    """
    Run Tesseract OCR on a preprocessed image and return the extracted
    raw text string.

    This function converts a NumPy image array (from OpenCV preprocessing)
    into a PIL Image, then passes it to Tesseract OCR for text recognition.

    Args:
        image (np.ndarray): A preprocessed image as a NumPy array.
                            Typically a grayscale or binary (black & white)
                            image from the preprocessor module.

    Returns:
        str: The raw text string extracted from the image by Tesseract.
             Returns an empty string if no text could be extracted.

    Raises:
        TypeError: If the input is not a valid NumPy array.
        pytesseract.TesseractNotFoundError: If Tesseract is not installed
            or not found in the system PATH.
    """

    # Validate input type
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected a NumPy array, got {type(image).__name__}. "
            "Please pass a preprocessed image from the preprocessor module."
        )

    # Convert the NumPy array to a PIL Image
    # pytesseract works best with PIL Image objects
    pil_image = Image.fromarray(image)

    # Configure Tesseract for optimal receipt text extraction
    # --psm 6: Assume a single uniform block of text
    #   This works well for receipts which are typically a vertical
    #   block of text. Other useful modes:
    #   --psm 3: Fully automatic page segmentation (default)
    #   --psm 4: Assume a single column of text
    custom_config = r"--psm 6"

    # Run Tesseract OCR on the PIL image
    try:
        raw_text = pytesseract.image_to_string(pil_image, config=custom_config)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR engine is not installed or not found in PATH. "
            "Install it via: brew install tesseract (Mac), "
            "sudo apt install tesseract-ocr (Linux), "
            "or download from https://github.com/UB-Mannheim/tesseract/wiki (Windows)."
        )

    # Strip leading/trailing whitespace and return
    return raw_text.strip()


def extract_text_with_confidence(image: np.ndarray) -> dict:
    """
    Run Tesseract OCR on a preprocessed image and return the extracted
    text along with word-level confidence scores.

    This function uses pytesseract.image_to_data() which provides
    detailed information about each recognized word, including its
    confidence score (0-100).

    Args:
        image (np.ndarray): A preprocessed image as a NumPy array.

    Returns:
        dict: A dictionary containing:
            - "raw_text" (str): The full extracted text string
            - "words" (list[dict]): List of word objects, each with:
                - "text" (str): The recognized word
                - "confidence" (float): Confidence score (0-100)
            - "average_confidence" (float): Average confidence across
              all recognized words (0-100)

    Raises:
        TypeError: If the input is not a valid NumPy array.
    """

    # Validate input type
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected a NumPy array, got {type(image).__name__}."
        )

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(image)

    # Configure Tesseract
    custom_config = r"--psm 6"

    # Get detailed OCR data including confidence scores
    # output_type=dict returns a dictionary with lists for each field
    try:
        ocr_data = pytesseract.image_to_data(
            pil_image,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR engine is not installed or not found in PATH. "
            "Install it via: brew install tesseract (Mac), "
            "sudo apt install tesseract-ocr (Linux), "
            "or download from https://github.com/UB-Mannheim/tesseract/wiki (Windows)."
        )

    # Build list of words with their confidence scores
    # Filter out empty text entries (Tesseract returns empty strings
    # for block/paragraph/line boundaries)
    words = []
    for i in range(len(ocr_data["text"])):
        word_text = ocr_data["text"][i].strip()
        confidence = int(ocr_data["conf"][i])

        # Skip empty entries and low-confidence noise (-1 means invalid)
        if word_text and confidence >= 0:
            words.append({
                "text": word_text,
                "confidence": confidence,
            })

    # Calculate average confidence across all valid words
    if words:
        avg_confidence = round(
            sum(w["confidence"] for w in words) / len(words), 1
        )
    else:
        avg_confidence = 0.0

    # Also get the full raw text via image_to_string
    raw_text = extract_text(image)

    return {
        "raw_text": raw_text,
        "words": words,
        "average_confidence": avg_confidence,
    }
