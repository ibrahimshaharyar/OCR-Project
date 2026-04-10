"""
preprocessor.py — Image Preprocessing Module

This module provides functions to prepare receipt/invoice images
for OCR text extraction. It uses OpenCV to apply a series of
image processing steps that improve Tesseract OCR accuracy:
    1. Convert to grayscale
    2. Apply denoising
    3. Apply binary thresholding (Otsu's method)

Usage:
    from app.preprocessor import preprocess_image
    cleaned_image = preprocess_image("path/to/receipt.jpg")
"""

import cv2
import numpy as np


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and apply preprocessing steps to improve
    OCR accuracy.

    Steps applied:
        1. Read the image from the given file path
        2. Convert from BGR color space to grayscale
        3. Apply non-local means denoising to reduce noise
        4. Apply Otsu's binary thresholding for clean black/white output

    Args:
        image_path (str): Absolute or relative path to the image file.

    Returns:
        np.ndarray: A preprocessed binary (black & white) image as a
                     NumPy array, ready for OCR processing.

    Raises:
        FileNotFoundError: If the image file does not exist at the given path.
        ValueError: If the file exists but cannot be read as a valid image.
    """

    # Step 1: Read the image from disk using OpenCV
    image = cv2.imread(image_path)

    # Validate that the image was loaded successfully
    if image is None:
        raise FileNotFoundError(
            f"Image not found or cannot be read: '{image_path}'. "
            "Please check the file path and ensure the file is a valid image."
        )

    # Step 2: Convert the image from BGR (OpenCV default) to grayscale
    # Grayscale reduces complexity and improves OCR performance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply non-local means denoising to remove noise
    # Parameters:
    #   - h=10: Filter strength. Higher value removes more noise but may
    #           also remove detail. 10 is a good balance for receipts.
    #   - templateWindowSize=7: Size of the patch used for denoising
    #   - searchWindowSize=21: Size of the area where the search is performed
    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Step 4: Apply Otsu's binary thresholding
    # Otsu's method automatically determines the optimal threshold value.
    # This converts the image to pure black and white, which helps
    # Tesseract distinguish text from background more accurately.
    # The first return value (threshold_value) is the computed threshold;
    # we only need the thresholded image.
    _, thresholded = cv2.threshold(
        denoised,
        0,                          # Initial threshold (ignored by Otsu)
        255,                        # Max pixel value (white)
        cv2.THRESH_BINARY + cv2.THRESH_OTSU  # Binary + automatic Otsu
    )

    return thresholded
