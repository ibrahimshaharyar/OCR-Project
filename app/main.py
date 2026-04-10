"""
main.py — FastAPI Application Entry Point

This module provides the REST API layer for the OCR receipt extraction tool.
It exposes endpoints for uploading receipt images and receiving structured
extraction results including vendor, date, total, raw text, and confidence.

Endpoints:
    GET  /              — Health check / welcome message
    POST /extract       — Upload an image, get extracted fields as JSON
    POST /extract/full  — Upload an image, get full results with raw text
                          and word-level confidence scores
    GET  /download/{filename} — Download a saved output file (JSON/CSV/Excel)

Usage:
    uvicorn app.main:app --reload
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import uuid
import shutil
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add the project root to the Python path so imports work
# when running via uvicorn from the receipt-ocr directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessor import preprocess_image
from app.ocr import extract_text, extract_text_with_confidence
from app.extractor import extract_fields
from utils.file_handler import save_results, save_to_json, save_to_csv, save_to_excel


# ================================================================
# Initialize the FastAPI application
# ================================================================

app = FastAPI(
    title="🧾 OCR Receipt Extraction API",
    description=(
        "Upload receipt or invoice images and extract structured data "
        "(vendor name, date, total amount) using Tesseract OCR. "
        "Returns JSON with confidence scores and supports Excel/CSV export."
    ),
    version="1.0.0",
)

# ================================================================
# CORS Middleware — allows the frontend to call this API
# ================================================================
# This is essential when the frontend runs on a different port
# (e.g., frontend on :3000, API on :8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow all origins (tighten in production)
    allow_credentials=True,
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"],         # Allow all headers
)

# ================================================================
# Directory for temporary file uploads
# ================================================================
TEMP_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "temp_uploads"
)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Output directory path
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs"
)


# ================================================================
# Helper function to save uploaded file temporarily
# ================================================================

def _save_upload_temp(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to a temporary directory and return its path.

    Uses a UUID prefix to prevent filename collisions when multiple
    users upload files simultaneously.

    Args:
        upload_file (UploadFile): The uploaded file from FastAPI.

    Returns:
        str: Absolute path to the saved temporary file.
    """
    # Generate a unique filename to prevent collisions
    unique_name = f"{uuid.uuid4().hex}_{upload_file.filename}"
    temp_path = os.path.join(TEMP_UPLOAD_DIR, unique_name)

    # Write the uploaded file contents to disk
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return temp_path


# ================================================================
# API Endpoints
# ================================================================

@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint. Returns a welcome message confirming
    the API is running.
    """
    return {
        "message": "🧾 OCR Receipt Extraction API is running!",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract": "Upload an image to extract receipt fields",
            "POST /extract/full": "Upload for full results with confidence data",
            "GET /download/{filename}": "Download saved output files",
        }
    }


@app.post("/extract", tags=["Extraction"])
async def extract_receipt(file: UploadFile = File(...)):
    """
    Extract vendor, date, and total from an uploaded receipt image.

    Accepts an image file (JPG, PNG, etc.), runs the full OCR pipeline,
    and returns the extracted fields as JSON.

    Args:
        file (UploadFile): The receipt/invoice image file to process.

    Returns:
        JSONResponse: Extracted fields with keys:
            - vendor (str | null)
            - date (str | null)
            - total (str | null)
            - confidence (float)
            - filename (str): base name used for saved files
    """

    # Validate that a file was uploaded
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Validate file type (basic check by extension)
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. "
                   f"Allowed: {', '.join(allowed_extensions)}"
        )

    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = _save_upload_temp(file)

        # Run the full pipeline
        # Step 1: Preprocess
        preprocessed = preprocess_image(temp_path)

        # Step 2: OCR with confidence
        ocr_result = extract_text_with_confidence(preprocessed)

        # Step 3: Extract fields
        result = extract_fields(ocr_result["raw_text"])
        result["confidence"] = ocr_result["average_confidence"]

        # Step 4: Save to all formats
        output_name = os.path.splitext(file.filename)[0]
        save_results(result, output_name)
        result["filename"] = output_name

        return JSONResponse(content=result, status_code=200)

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/extract/full", tags=["Extraction"])
async def extract_receipt_full(file: UploadFile = File(...)):
    """
    Extract receipt fields with full detail — including raw OCR text
    and word-level confidence scores.

    This endpoint returns everything needed for the frontend to display:
    - Extracted fields (vendor, date, total)
    - Full raw OCR text
    - Per-word confidence scores (for highlighting in UI)
    - Overall average confidence

    Args:
        file (UploadFile): The receipt/invoice image file to process.

    Returns:
        JSONResponse: Complete extraction result including raw text
                      and word-level confidence data.
    """

    # Validate that a file was uploaded
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. "
                   f"Allowed: {', '.join(allowed_extensions)}"
        )

    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = _save_upload_temp(file)

        # Run the full pipeline
        preprocessed = preprocess_image(temp_path)
        ocr_result = extract_text_with_confidence(preprocessed)
        fields = extract_fields(ocr_result["raw_text"])

        # Save to all formats
        output_name = os.path.splitext(file.filename)[0]
        save_results(fields, output_name)

        # Build the full response with all data for the frontend
        full_response = {
            "fields": fields,
            "raw_text": ocr_result["raw_text"],
            "words": ocr_result["words"],
            "average_confidence": ocr_result["average_confidence"],
            "filename": output_name,
        }

        return JSONResponse(content=full_response, status_code=200)

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/download/{filename}", tags=["Download"])
async def download_file(filename: str, format: str = "json"):
    """
    Download a previously saved output file.

    Supports downloading in JSON, CSV, or Excel format.

    Args:
        filename (str): Base filename (without extension) of the output.
        format (str): File format to download. Options: json, csv, excel.
                      Defaults to "json".

    Returns:
        FileResponse: The requested file as a download attachment.
    """

    # Map format parameter to file extension and media type
    format_map = {
        "json": (".json", "application/json"),
        "csv": (".csv", "text/csv"),
        "excel": (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "xlsx": (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    }

    if format.lower() not in format_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: '{format}'. Options: json, csv, excel"
        )

    ext, media_type = format_map[format.lower()]
    filepath = os.path.join(OUTPUT_DIR, f"{filename}{ext}")

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"File not found: '{filename}{ext}'. "
                   "Process a receipt first via /extract."
        )

    return FileResponse(
        path=filepath,
        media_type=media_type,
        filename=f"{filename}{ext}",
    )
