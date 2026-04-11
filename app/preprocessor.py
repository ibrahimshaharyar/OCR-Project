"""
preprocessor.py — Image Preprocessing Module (Enhanced)

This module provides an advanced image preprocessing pipeline
for receipt/invoice images to maximize Tesseract OCR accuracy.

Processing pipeline:
    1. Load and validate image
    2. Resize/upscale to optimal DPI (~300 DPI equivalent)
    3. Convert to grayscale
    4. Apply CLAHE (adaptive contrast enhancement)
    5. Apply sharpening (unsharp mask)
    6. Apply denoising (bilateral filter — preserves edges better)
    7. Apply adaptive thresholding (handles uneven lighting)
    8. Apply morphological cleanup (close small gaps in text)

Usage:
    from app.preprocessor import preprocess_image
    cleaned_image = preprocess_image("path/to/receipt.jpg")
"""

import cv2
import numpy as np


def _resize_for_ocr(image: np.ndarray, target_height: int = 2000) -> np.ndarray:
    """
    Resize the image so its height is at least target_height pixels.

    Tesseract performs best on images with ~300 DPI. Most receipt photos
    from phones are too small. Upscaling improves character recognition
    significantly.

    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        target_height (int): Minimum height in pixels. Default 2000px
                             which approximates 300 DPI for most receipts.

    Returns:
        np.ndarray: Resized image (only upscales, never downscales).
    """
    h, w = image.shape[:2]

    # Only upscale — never shrink the image
    if h >= target_height:
        return image

    # Calculate scale factor to reach target height
    scale = target_height / h
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Use INTER_CUBIC for upscaling — best quality for text
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE improves contrast locally rather than globally, which is
    critical for receipts with uneven lighting (e.g., shadows, folds,
    or faded thermal print areas).

    Args:
        gray (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(
        clipLimit=2.0,        # Limits contrast amplification (prevents noise boost)
        tileGridSize=(8, 8)   # Divides image into 8x8 tiles for local processing
    )
    return clahe.apply(gray)


def _sharpen(image: np.ndarray) -> np.ndarray:
    """
    Apply unsharp masking to sharpen text edges.

    Sharpening makes character boundaries crisper, which directly
    improves Tesseract's ability to distinguish similar characters
    (e.g., 'l' vs '1', 'O' vs '0').

    Technique: subtract a Gaussian-blurred version from the original
    and add the difference back with a weight.

    Args:
        image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Sharpened image.
    """
    # Create a blurred copy
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)

    # Unsharp mask: original + weight * (original - blurred)
    # alpha=1.5: how much of the original to keep
    # beta=-0.5: how much of the blur to subtract
    # gamma=0: brightness offset
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    return sharpened


def _adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding instead of global Otsu.

    Adaptive thresholding calculates a threshold for each small region
    of the image, making it far more robust for:
    - Thermal receipts with fading
    - Photos with shadows or uneven lighting
    - Creased or curved receipts

    Args:
        image (np.ndarray): Denoised grayscale image.

    Returns:
        np.ndarray: Binary (black & white) image.
    """
    binary = cv2.adaptiveThreshold(
        image,
        255,                                  # Max value (white)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # Gaussian-weighted neighbourhood
        cv2.THRESH_BINARY,                    # Output is binary
        21,                                   # Block size (neighbourhood size)
        10                                    # Constant subtracted from mean
    )
    return binary


def _morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean up the binary image.

    - Closing: fills small holes/gaps inside characters (broken 'e', 'a')
    - This helps Tesseract recognize characters that got fragmented
      during thresholding.

    Args:
        binary (np.ndarray): Binary (B&W) image.

    Returns:
        np.ndarray: Cleaned binary image.
    """
    # Small rectangular kernel — just enough to close 1-2 pixel gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Closing = dilation → erosion (fills small gaps without growing text)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and apply the full enhanced preprocessing
    pipeline to maximize OCR accuracy.

    Pipeline:
        1. Read image from disk
        2. Upscale to ~300 DPI equivalent (if smaller)
        3. Convert BGR → Grayscale
        4. CLAHE adaptive contrast enhancement
        5. Unsharp mask sharpening
        6. Bilateral denoising (preserves edges, removes noise)
        7. Adaptive thresholding (handles uneven lighting)
        8. Morphological closing (fills broken characters)

    Args:
        image_path (str): Absolute or relative path to the image file.

    Returns:
        np.ndarray: A preprocessed binary (black & white) image as a
                     NumPy array, ready for OCR processing.

    Raises:
        FileNotFoundError: If the image file does not exist at the given path.
    """

    # Step 1: Read the image from disk
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found or cannot be read: '{image_path}'. "
            "Please check the file path and ensure the file is a valid image."
        )

    # Step 2: Upscale small images for better OCR
    # Tesseract works best at ~300 DPI; most phone photos are lower
    image = _resize_for_ocr(image, target_height=2000)

    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: CLAHE — adaptive contrast enhancement
    # Fixes faded thermal receipts and uneven lighting
    enhanced = _apply_clahe(gray)

    # Step 5: Sharpen text edges
    # Makes character boundaries crisper for Tesseract
    sharpened = _sharpen(enhanced)

    # Step 6: Bilateral filter denoising
    # Unlike fastNlMeansDenoising, bilateral filter preserves edges
    # while smoothing noise — critical for text preservation
    denoised = cv2.bilateralFilter(
        sharpened,
        d=9,              # Diameter of pixel neighbourhood
        sigmaColor=75,    # Filter sigma in the color space
        sigmaSpace=75     # Filter sigma in the coordinate space
    )

    # Step 7: Adaptive thresholding
    # Handles uneven lighting far better than global Otsu
    binary = _adaptive_threshold(denoised)

    # Step 8: Morphological cleanup
    # Close small gaps in characters from thresholding
    cleaned = _morphological_cleanup(binary)

    return cleaned
