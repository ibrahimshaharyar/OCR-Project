"""
test_extractor.py — Unit Tests for Field Extraction Logic

Tests the regex-based extraction of vendor, date, and total fields
from raw OCR text. These tests do NOT require Tesseract to be installed
since they test the extraction logic directly with sample text strings.

Usage:
    python -m pytest tests/test_extractor.py -v
"""

import sys
import os

# Add project root to path so imports work when running pytest directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.extractor import extract_vendor, extract_date, extract_total, extract_fields


# ==============================================================
# Vendor Extraction Tests
# ==============================================================

class TestExtractVendor:
    """Tests for the extract_vendor() function."""

    def test_vendor_from_first_line(self):
        """Vendor should be the first non-empty, non-numeric line."""
        text = "WALMART SUPERCENTER\n123 Main St\nDate: 01/15/2024"
        assert extract_vendor(text) == "WALMART SUPERCENTER"

    def test_vendor_skips_empty_lines(self):
        """Should skip blank lines at the top of the text."""
        text = "\n\n\nTARGET STORE\n456 Oak Ave"
        assert extract_vendor(text) == "TARGET STORE"

    def test_vendor_skips_numeric_lines(self):
        """Should skip lines that are only numbers or symbols."""
        text = "12345\n---\nBEST BUY\nElectronics"
        assert extract_vendor(text) == "BEST BUY"

    def test_vendor_returns_none_for_empty_input(self):
        """Should return None when input is empty."""
        assert extract_vendor("") is None

    def test_vendor_returns_none_for_only_numbers(self):
        """Should return None when text has only numbers."""
        text = "12345\n67890\n---"
        assert extract_vendor(text) is None

    def test_vendor_with_special_characters(self):
        """Vendor names with ampersands or apostrophes should work."""
        text = "MCDONALD'S RESTAURANT\nOrder #1234"
        assert extract_vendor(text) == "MCDONALD'S RESTAURANT"


# ==============================================================
# Date Extraction Tests
# ==============================================================

class TestExtractDate:
    """Tests for the extract_date() function."""

    def test_date_dd_mm_yyyy_slash(self):
        """Should match DD/MM/YYYY format."""
        text = "Receipt\nDate: 15/01/2024\nTotal: $10"
        assert extract_date(text) == "15/01/2024"

    def test_date_mm_dd_yyyy_dash(self):
        """Should match MM-DD-YYYY format."""
        text = "Store\nDate: 01-15-2024\nItems"
        assert extract_date(text) == "01-15-2024"

    def test_date_yyyy_mm_dd_iso(self):
        """Should match YYYY-MM-DD (ISO) format."""
        text = "Invoice\n2024-01-15\nAmount: $50"
        assert extract_date(text) == "2024-01-15"

    def test_date_month_name_full(self):
        """Should match 'January 15, 2024' format."""
        text = "Store Name\nJanuary 15, 2024\nItem: $5"
        assert extract_date(text) == "January 15, 2024"

    def test_date_month_name_abbreviated(self):
        """Should match 'Jan 15, 2024' format."""
        text = "Store\nDate: Jan 15, 2024\nTotal"
        assert extract_date(text) == "Jan 15, 2024"

    def test_date_day_month_name_year(self):
        """Should match '15 January 2024' format."""
        text = "Store\n15 January 2024\nTotal: $20"
        assert extract_date(text) == "15 January 2024"

    def test_date_day_month_abbrev_year(self):
        """Should match '15 Jan 2024' format."""
        text = "Store\n15 Jan 2024\nTotal"
        assert extract_date(text) == "15 Jan 2024"

    def test_date_case_insensitive(self):
        """Date matching should be case-insensitive."""
        text = "Store\nFEBRUARY 20, 2024\nTotal"
        assert extract_date(text) == "FEBRUARY 20, 2024"

    def test_date_returns_none_when_missing(self):
        """Should return None when no date is found."""
        text = "Store Name\nItem: $5\nTotal: $5"
        assert extract_date(text) is None

    def test_date_returns_none_for_empty(self):
        """Should return None for empty input."""
        assert extract_date("") is None

    def test_date_month_name_no_comma(self):
        """Should match 'March 03 2025' without comma."""
        text = "Store\nMarch 03 2025\nTotal"
        assert extract_date(text) == "March 03 2025"


# ==============================================================
# Total Extraction Tests
# ==============================================================

class TestExtractTotal:
    """Tests for the extract_total() function."""

    def test_total_with_keyword(self):
        """Should extract amount after 'Total' keyword."""
        text = "Item: $5\nTax: $0.40\nTotal: $5.40"
        assert extract_total(text) == "5.40"

    def test_total_with_dollar_sign(self):
        """Should handle dollar sign before the amount."""
        text = "Subtotal: $10\nTotal: $12.99"
        assert extract_total(text) == "12.99"

    def test_grand_total_keyword(self):
        """Should match 'Grand Total' keyword."""
        text = "Subtotal: $30\nTax: $2.40\nGrand Total: $32.40"
        assert extract_total(text) == "32.40"

    def test_amount_due_keyword(self):
        """Should match 'Amount Due' keyword."""
        text = "Invoice\nServices: $100\nAmount Due: $100.00"
        assert extract_total(text) == "100.00"

    def test_balance_due_keyword(self):
        """Should match 'Balance Due' keyword."""
        text = "Invoice\nBalance Due: $250.00"
        assert extract_total(text) == "250.00"

    def test_total_due_keyword(self):
        """Should match 'Total Due' keyword."""
        text = "Bill\nTotal Due $75.50"
        assert extract_total(text) == "75.50"

    def test_total_case_insensitive(self):
        """Total matching should be case-insensitive."""
        text = "Items\nTOTAL: $99.99"
        assert extract_total(text) == "99.99"

    def test_total_returns_last_match(self):
        """When multiple totals exist, should return the last one (grand total)."""
        text = "Subtotal: $20.00\nTax: $1.60\nTotal: $21.60"
        result = extract_total(text)
        assert result == "21.60"

    def test_total_returns_none_when_missing(self):
        """Should return None when no total is found."""
        text = "Store Name\nItem 1\nItem 2"
        assert extract_total(text) is None

    def test_total_returns_none_for_empty(self):
        """Should return None for empty input."""
        assert extract_total("") is None

    def test_total_without_cents(self):
        """Should handle whole numbers without decimal."""
        text = "Total: $50"
        assert extract_total(text) == "50"

    def test_total_with_euro_sign(self):
        """Should handle euro currency symbol."""
        text = "Total: €25.99"
        assert extract_total(text) == "25.99"


# ==============================================================
# Full Extraction Pipeline Tests
# ==============================================================

class TestExtractFields:
    """Tests for the extract_fields() orchestrator function."""

    def test_full_extraction(self):
        """Should extract all three fields from a complete receipt."""
        text = (
            "WALMART SUPERCENTER\n"
            "123 Main Street\n"
            "Date: 01/15/2024\n"
            "Milk: $3.99\n"
            "Bread: $2.49\n"
            "Total: $6.48"
        )
        result = extract_fields(text)
        assert result["vendor"] == "WALMART SUPERCENTER"
        assert result["date"] == "01/15/2024"
        assert result["total"] == "6.48"

    def test_all_fields_none_for_empty_input(self):
        """Should return all None values for empty string input."""
        result = extract_fields("")
        assert result == {"vendor": None, "date": None, "total": None}

    def test_all_fields_none_for_none_input(self):
        """Should return all None values for None input."""
        result = extract_fields(None)
        assert result == {"vendor": None, "date": None, "total": None}

    def test_partial_extraction_missing_date(self):
        """Should handle missing date gracefully."""
        text = "COSTCO\nItem: $10\nTotal: $10.00"
        result = extract_fields(text)
        assert result["vendor"] == "COSTCO"
        assert result["date"] is None
        assert result["total"] == "10.00"

    def test_partial_extraction_missing_total(self):
        """Should handle missing total gracefully."""
        text = "TRADER JOE'S\n01/20/2024\nBananas $1.29"
        result = extract_fields(text)
        assert result["vendor"] == "TRADER JOE'S"
        assert result["date"] == "01/20/2024"
        assert result["total"] is None

    def test_result_is_dict_with_correct_keys(self):
        """Result should always be a dict with vendor, date, total keys."""
        result = extract_fields("any text")
        assert isinstance(result, dict)
        assert "vendor" in result
        assert "date" in result
        assert "total" in result

    def test_realistic_receipt_text(self):
        """Test with realistic OCR output from a receipt."""
        text = (
            "FRESH MART GROCERY\n"
            "123 Market Street\n"
            "New York, NY 10001\n"
            "Tel: (555) 123-4567\n"
            "\n"
            "Date: 03/15/2024\n"
            "\n"
            "Organic Milk      $4.99\n"
            "Wheat Bread       $3.49\n"
            "Fresh Eggs        $5.99\n"
            "\n"
            "Subtotal:        $14.47\n"
            "Tax (8%):         $1.16\n"
            "Total:           $15.63\n"
            "\n"
            "Thank you for shopping!"
        )
        result = extract_fields(text)
        assert result["vendor"] == "FRESH MART GROCERY"
        assert result["date"] == "03/15/2024"
        assert result["total"] == "15.63"
