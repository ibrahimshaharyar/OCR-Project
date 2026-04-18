"""
preprocessor.py — Image Preprocessing Module (Dual-Pipeline)

This module provides TWO preprocessing pipelines and automatically
picks the one that produces the best OCR result for each image:

Pipeline A (Simple — faithful to original):
    1. Load image → Upscale → Grayscale
    2. Light denoising (non-local means)
    3. Otsu binary thresholding

Pipeline B (Enhanced — aggressive cleanup):
    1. Load image → Upscale → Grayscale
    2. CLAHE contrast → Gentle sharpening
    3. Bilateral denoising → Adaptive thresholding
    4. Morphological closing

The dual approach ensures that each receipt image gets whichever
pipeline best preserves its text — simple receipts get Pipeline A,
challenging ones (shadows, fading) get Pipeline B.

Usage:
    from app.preprocessor import preprocess_image, preprocess_dual
    # Single best result:
    cleaned_image = preprocess_image("path/to/receipt.jpg")
    # Both pipelines (for comparison in OCR module):
    simple, enhanced = preprocess_dual("path/to/receipt.jpg")
"""

import cv2
import numpy as np


# ============================================
# Shared Helper Functions
# ============================================

def _load_and_validate(image_path: str) -> np.ndarray:
    """
    Load an image from disk and validate it.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The loaded BGR image.

    Raises:
        FileNotFoundError: If the image doesn't exist or can't be read.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Image not found or cannot be read: '{image_path}'. "
            "Please check the file path and ensure the file is a valid image."
        )
    return image


def _resize_for_ocr(image: np.ndarray, target_height: int = 2000) -> np.ndarray:
    """
    Upscale small images to improve OCR accuracy.
    Tesseract works best at ~300 DPI. Only upscales, never shrinks.

    Args:
        image (np.ndarray): Input image.
        target_height (int): Minimum height in pixels.

    Returns:
        np.ndarray: Resized image.
    """
    h, w = image.shape[:2]
    if h >= target_height:
        return image

    scale = target_height / h
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


# ============================================
# Pipeline A: Simple (faithful to original text)
# ============================================

def _preprocess_simple(gray: np.ndarray) -> np.ndarray:
    """
    Simple preprocessing pipeline — minimal processing to preserve
    the original text as faithfully as possible.

    Steps:
        1. Non-local means denoising (gentle)
        2. Otsu binary thresholding (global)

    This works best for:
        - Clean, well-lit receipt photos
        - Printed text on white backgrounds
        - High-resolution images

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Binary image.
    """
    # Gentle denoising — preserves text detail
    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Global Otsu thresholding — simple but faithful
    _, binary = cv2.threshold(
        denoised, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


# ============================================
# Pipeline B: Enhanced (handles difficult images)
# ============================================

def _preprocess_enhanced(gray: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing pipeline — more aggressive processing
    to handle challenging receipt images.

    Steps:
        1. CLAHE contrast enhancement (gentle clipLimit)
        2. Light sharpening (reduced weight to avoid artifacts)
        3. Bilateral denoising (edge-preserving)
        4. Adaptive thresholding (handles uneven lighting)
        5. Morphological closing (fixes broken characters)

    This works best for:
        - Faded thermal receipts
        - Photos with shadows/uneven lighting
        - Low-contrast images

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Binary image.
    """
    # CLAHE — gentle contrast boost (lower clipLimit to avoid noise amp)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light sharpening — reduced weight to prevent halo artifacts
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)

    # Bilateral denoising — preserves edges while removing noise
    denoised = cv2.bilateralFilter(
        sharpened, d=9, sigmaColor=50, sigmaSpace=50
    )

    # Adaptive thresholding — local thresholds for uneven lighting
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )

    # Morphological closing — fill tiny gaps in broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned


# ============================================
# Public API
# ============================================

def preprocess_dual(image_path: str) -> tuple:
    """
    Load an image and run BOTH preprocessing pipelines.

    Returns both results so the OCR module can compare them
    and pick the one with better actual text quality.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (simple_result, enhanced_result) — both np.ndarray
    """
    image = _load_and_validate(image_path)
    image = _resize_for_ocr(image, target_height=2000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    simple = _preprocess_simple(gray)
    enhanced = _preprocess_enhanced(gray)

    return simple, enhanced


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image and apply the simple preprocessing pipeline.

    This function is kept for backward compatibility with run.py
    and any code that expects a single image result. The OCR module
    now uses preprocess_dual() for the dual-pipeline strategy.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed binary image.
    """
    image = _load_and_validate(image_path)
    image = _resize_for_ocr(image, target_height=2000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return _preprocess_simple(gray)
