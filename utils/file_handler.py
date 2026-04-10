"""
file_handler.py — File Export Utilities

This module provides functions to save OCR extraction results
to JSON and CSV files in the outputs/ directory.

It supports saving individual result dictionaries as well as
a combined save function that exports to both formats at once.

Usage:
    from utils.file_handler import save_results
    data = {"vendor": "WALMART", "date": "01/15/2024", "total": "45.99"}
    save_results(data, "receipt_001")
    # Creates: outputs/receipt_001.json and outputs/receipt_001.csv
"""

import json
import os

import pandas as pd


# Define the output directory path (relative to project root)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def _ensure_output_dir() -> None:
    """
    Ensure the outputs/ directory exists. Creates it if it doesn't.

    This is called internally before any file write operation to
    prevent FileNotFoundError.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_to_json(data: dict, filename: str) -> str:
    """
    Save extraction results as a JSON file in the outputs/ directory.

    Args:
        data (dict): The extraction result dictionary containing
                     keys like "vendor", "date", and "total".
        filename (str): Base filename without extension
                        (e.g., "receipt_001").

    Returns:
        str: The full path to the saved JSON file.

    Raises:
        TypeError: If data is not a dictionary.
        ValueError: If filename is empty.
    """

    # Validate inputs
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary, got {type(data).__name__}.")
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty.")

    # Ensure the output directory exists
    _ensure_output_dir()

    # Build the full file path
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.json")

    # Write the JSON file with pretty formatting
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"✅ JSON saved: {filepath}")
    return filepath


def save_to_csv(data: dict, filename: str) -> str:
    """
    Save extraction results as a CSV file in the outputs/ directory.

    Creates a single-row CSV with column headers matching the
    dictionary keys (vendor, date, total).

    Args:
        data (dict): The extraction result dictionary containing
                     keys like "vendor", "date", and "total".
        filename (str): Base filename without extension
                        (e.g., "receipt_001").

    Returns:
        str: The full path to the saved CSV file.

    Raises:
        TypeError: If data is not a dictionary.
        ValueError: If filename is empty.
    """

    # Validate inputs
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary, got {type(data).__name__}.")
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty.")

    # Ensure the output directory exists
    _ensure_output_dir()

    # Build the full file path
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.csv")

    # Create a single-row DataFrame from the dictionary
    # Wrapping data in a list creates one row with dict keys as columns
    df = pd.DataFrame([data])

    # Write to CSV without the pandas index column
    df.to_csv(filepath, index=False, encoding="utf-8")

    print(f"✅ CSV saved: {filepath}")
    return filepath


def save_results(data: dict, filename: str) -> dict:
    """
    Save extraction results to JSON, CSV, and Excel formats.

    This is a convenience function that calls save_to_json(),
    save_to_csv(), and save_to_excel() with the same data and filename.

    Args:
        data (dict): The extraction result dictionary containing
                     keys like "vendor", "date", and "total".
        filename (str): Base filename without extension
                        (e.g., "receipt_001").

    Returns:
        dict: A dictionary with the paths to all saved files:
              {"json": "...", "csv": "...", "excel": "..."}

    Example:
        >>> result = {"vendor": "WALMART", "date": "01/15/2024", "total": "45.99"}
        >>> paths = save_results(result, "receipt_001")
    """

    # Save in all formats
    json_path = save_to_json(data, filename)
    csv_path = save_to_csv(data, filename)
    excel_path = save_to_excel(data, filename)

    return {"json": json_path, "csv": csv_path, "excel": excel_path}


def save_to_excel(data: dict, filename: str) -> str:
    """
    Save extraction results as an Excel (.xlsx) file in the outputs/ directory.

    Creates a single-row Excel spreadsheet with column headers matching
    the dictionary keys. Uses openpyxl as the engine for .xlsx support.

    Args:
        data (dict): The extraction result dictionary containing
                     keys like "vendor", "date", and "total".
        filename (str): Base filename without extension
                        (e.g., "receipt_001").

    Returns:
        str: The full path to the saved Excel file.

    Raises:
        TypeError: If data is not a dictionary.
        ValueError: If filename is empty.
    """

    # Validate inputs
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary, got {type(data).__name__}.")
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty.")

    # Ensure the output directory exists
    _ensure_output_dir()

    # Build the full file path
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.xlsx")

    # Create a single-row DataFrame from the dictionary
    df = pd.DataFrame([data])

    # Write to Excel without the pandas index column
    # engine='openpyxl' is required for .xlsx format
    df.to_excel(filepath, index=False, engine="openpyxl")

    print(f"✅ Excel saved: {filepath}")
    return filepath

