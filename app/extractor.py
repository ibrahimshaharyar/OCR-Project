"""
extractor.py — Field Extraction Module

This module extracts structured data from raw OCR text using regex patterns.
It identifies three key fields from receipt/invoice text:
    1. Vendor name — the first meaningful non-empty line
    2. Date — supports multiple date formats
    3. Total amount — looks for total-related keywords followed by a number

All regex patterns are case-insensitive.
All extraction functions return None for fields they cannot find,
ensuring the tool never crashes on missing data.

Usage:
    from app.extractor import extract_fields
    result = extract_fields(raw_ocr_text)
    # result = {"vendor": "WALMART", "date": "01/15/2024", "total": "45.99"}
"""

import re


def extract_vendor(text: str) -> str:
    """
    Extract the vendor/store name from receipt text.

    Strategy: The vendor name is typically the first non-empty line
    on a receipt that contains meaningful text (not just numbers,
    dashes, or special characters).

    Args:
        text (str): Raw OCR text extracted from a receipt image.

    Returns:
        str or None: The vendor name string, or None if no valid
                     vendor line could be identified.
    """

    # Split text into individual lines
    lines = text.split("\n")

    for line in lines:
        # Strip whitespace from each line
        cleaned = line.strip()

        # Skip empty lines
        if not cleaned:
            continue

        # Skip lines that are only numbers, symbols, or whitespace
        # A valid vendor name should contain at least some letters
        if re.match(r"^[\d\W]+$", cleaned):
            continue

        # Skip very short lines (likely noise or partial characters)
        if len(cleaned) < 2:
            continue

        # Return the first line that passes all filters
        return cleaned

    # No valid vendor line found
    return None


def extract_date(text: str) -> str:
    """
    Extract a date from receipt text using multiple format patterns.

    Supported date formats:
        - DD/MM/YYYY  (e.g., 15/01/2024)
        - MM-DD-YYYY  (e.g., 01-15-2024)
        - YYYY-MM-DD  (e.g., 2024-01-15)
        - Month name variants:
            - January 15, 2024
            - 15 January 2024
            - Jan 15, 2024
            - 15 Jan 2024
            - January 15 2024 (without comma)

    Args:
        text (str): Raw OCR text extracted from a receipt image.

    Returns:
        str or None: The first date string found in the text,
                     or None if no date could be identified.
    """

    # Define regex patterns for each supported date format
    # Order matters — more specific patterns should come first
    date_patterns = [
        # YYYY-MM-DD (ISO format, e.g., 2024-01-15)
        r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",

        # DD/MM/YYYY or MM/DD/YYYY (e.g., 15/01/2024 or 01/15/2024)
        r"\b(\d{1,2}/\d{1,2}/\d{4})\b",

        # MM-DD-YYYY or DD-MM-YYYY (e.g., 01-15-2024)
        r"\b(\d{1,2}-\d{1,2}-\d{4})\b",

        # Month name DD, YYYY (e.g., January 15, 2024 or Jan 15, 2024)
        r"\b((?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"\s+\d{1,2},?\s+\d{4})\b",

        # DD Month name YYYY (e.g., 15 January 2024 or 15 Jan 2024)
        r"\b(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"\s+\d{4})\b",
    ]

    # Try each pattern against the text (case-insensitive)
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # No date found in any format
    return None


def extract_total(text: str) -> str:
    """
    Extract the total amount from receipt text.

    Strategy: Look for keywords commonly associated with the final
    total on a receipt, followed by a currency symbol (optional)
    and a numeric value.

    Keywords searched (case-insensitive):
        - total
        - grand total
        - amount due
        - balance due
        - total due
        - amount

    Args:
        text (str): Raw OCR text extracted from a receipt image.

    Returns:
        str or None: The total amount as a string (e.g., "45.99"),
                     or None if no total could be identified.
    """

    # Pattern breakdown:
    #   (grand\s+total|total\s+due|amount\s+due|balance\s+due|total|amount)
    #       — matches the keyword (multi-word keywords listed first to
    #         prevent partial matches)
    #   [:\s]*   — optional colon or whitespace after keyword
    #   [\$€£]?  — optional currency symbol
    #   \s*      — optional whitespace
    #   (\d+[.,]?\d{0,2})  — the numeric amount (e.g., 45.99, 1,234.56)
    total_pattern = (
        r"(?:grand\s*total|total\s*due|amount\s*due|balance\s*due|total|amount)"
        r"[:\s]*"
        r"[\$€£]?\s*"
        r"(\d{1,},?\d*\.?\d{0,2})"
    )

    # Search the text for the total pattern (case-insensitive)
    matches = re.findall(total_pattern, text, re.IGNORECASE)

    if matches:
        # Return the LAST match — on receipts, the final "Total" line
        # is typically the grand total (subtotals appear earlier)
        return matches[-1].strip()

    # No total amount found
    return None


def extract_fields(text: str) -> dict:
    """
    Extract all structured fields from raw OCR receipt text.

    This is the main orchestrator function that calls individual
    extraction functions for each field and combines the results
    into a single dictionary.

    Args:
        text (str): Raw OCR text extracted from a receipt image.

    Returns:
        dict: A dictionary with the following keys:
            - "vendor" (str or None): The vendor/store name
            - "date" (str or None): The transaction date
            - "total" (str or None): The total amount

    Example:
        >>> extract_fields("WALMART\\nDate: 01/15/2024\\nTotal: $45.99")
        {"vendor": "WALMART", "date": "01/15/2024", "total": "45.99"}

        >>> extract_fields("")
        {"vendor": None, "date": None, "total": None}
    """

    # Handle None or non-string input gracefully
    if not text or not isinstance(text, str):
        return {"vendor": None, "date": None, "total": None}

    # Extract each field independently
    vendor = extract_vendor(text)
    date = extract_date(text)
    total = extract_total(text)

    # Return the structured result dictionary
    return {
        "vendor": vendor,
        "date": date,
        "total": total,
    }
