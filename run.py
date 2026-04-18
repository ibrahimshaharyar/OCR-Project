"""
run.py — CLI Entry Point

This is the command-line interface for the OCR receipt extraction tool.
It accepts an image file path as an argument, runs the full extraction
pipeline, and outputs the results to the terminal and to files.

Usage:
    python run.py samples/sample_receipt.jpg
    python run.py /path/to/receipt.png --output my_receipt

What it does:
    1. Preprocesses the image (grayscale, denoise, threshold)
    2. Runs Tesseract OCR to extract text
    3. Uses regex to extract vendor, date, and total
    4. Prints the result as formatted JSON in the terminal
    5. Saves JSON, CSV, and Excel files to the outputs/ folder
"""

import argparse
import json
import os
import sys

# Add the project root to the Python path so imports work
# when running this script directly from the command line
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.preprocessor import preprocess_dual
from app.ocr import extract_best_from_dual
from app.extractor import extract_fields
from utils.file_handler import save_results


def run_pipeline(image_path: str, output_name: str = None) -> dict:
    """
    Run the full OCR extraction pipeline on a single image.

    Pipeline steps:
        1. Preprocess image (grayscale → denoise → threshold)
        2. Extract raw text via Tesseract OCR
        3. Parse vendor, date, and total from the text
        4. Save results to JSON, CSV, and Excel files

    Args:
        image_path (str): Path to the receipt/invoice image file.
        output_name (str, optional): Base filename for output files.
                                     If not provided, derives from the
                                     input image filename.

    Returns:
        dict: The extraction result with keys: vendor, date, total.
    """

    # Validate the image path exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: '{image_path}'")
        sys.exit(1)

    # Generate output filename from input image name if not provided
    if not output_name:
        # Extract filename without extension: "receipt.jpg" → "receipt"
        output_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"📄 Processing: {image_path}")
    print("-" * 50)

    # Step 1: Preprocess the image with BOTH pipelines
    print("🔧 Step 1: Preprocessing image (dual pipeline)...")
    simple_img, enhanced_img = preprocess_dual(image_path)
    print("   ✅ Simple pipeline: grayscale → denoise → Otsu")
    print("   ✅ Enhanced pipeline: CLAHE → sharpen → adaptive threshold")

    # Step 2: Run OCR on both, pick best by quality score
    print("🔍 Step 2: Running OCR (comparing both pipelines)...")
    ocr_result = extract_best_from_dual(simple_img, enhanced_img)
    raw_text = ocr_result["raw_text"]
    avg_confidence = ocr_result["average_confidence"]
    pipeline_used = ocr_result.get("pipeline_used", "unknown")
    print(f"   ✅ Text extracted ({len(raw_text)} characters)")
    print(f"   📊 Average confidence: {avg_confidence}%")
    print(f"   🏆 Best pipeline: {pipeline_used}")

    # Show the raw extracted text
    print("\n📝 Raw OCR Text:")
    print("-" * 50)
    print(raw_text if raw_text else "(No text detected)")
    print("-" * 50)

    # Step 3: Extract structured fields
    print("\n🎯 Step 3: Extracting fields...")
    result = extract_fields(raw_text)

    # Add confidence score to the result
    result["confidence"] = avg_confidence

    # Step 4: Save results to files
    print("💾 Step 4: Saving results...")
    saved_paths = save_results(result, output_name)

    # Print the final JSON result
    print("\n" + "=" * 50)
    print("📋 EXTRACTION RESULT:")
    print("=" * 50)
    print(json.dumps(result, indent=4))
    print("=" * 50)

    # Print saved file paths
    print(f"\n📁 Files saved to:")
    for format_name, path in saved_paths.items():
        print(f"   → {format_name.upper()}: {path}")

    return result


def main():
    """
    Parse command-line arguments and run the extraction pipeline.

    CLI Arguments:
        image_path (required): Path to the receipt/invoice image
        --output, -o (optional): Base name for output files
    """

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="🧾 OCR Receipt Extraction Tool — Extract vendor, "
                    "date, and total from receipt images.",
        epilog="Example: python run.py samples/sample_receipt.jpg"
    )

    # Required argument: path to the image file
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the receipt or invoice image file (JPG, PNG, etc.)"
    )

    # Optional argument: custom output filename
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Base filename for the output files (without extension). "
             "Defaults to the input image filename."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the pipeline
    run_pipeline(args.image_path, args.output)


if __name__ == "__main__":
    main()
