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
        raise pytesseract.TesseractNotFoundError(
            "Tesseract OCR engine is not installed or not found in PATH. "
            "Install it via: brew install tesseract (Mac), "
            "sudo apt install tesseract-ocr (Linux), "
            "or download from https://github.com/UB-Mannheim/tesseract/wiki (Windows)."
        )

    # Strip leading/trailing whitespace and return
    return raw_text.strip()
