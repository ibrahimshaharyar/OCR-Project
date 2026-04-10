"""
generate_sample.py — Sample Receipt Image Generator

Creates a realistic-looking receipt image programmatically using Pillow.
This ensures the text is clean and OCR-readable for testing purposes.

Usage:
    python generate_sample.py
"""

from PIL import Image, ImageDraw, ImageFont
import os


def create_sample_receipt(output_path: str = "samples/sample_receipt.jpg"):
    """Generate a sample receipt image with readable text."""

    # Create a white canvas (receipt dimensions)
    width, height = 400, 700
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Try to use a monospace font, fall back to default
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Courier.dfont", 22)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Courier.dfont", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Courier.dfont", 14)
    except (IOError, OSError):
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        except (IOError, OSError):
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

    # Receipt content lines
    y = 30
    lines = [
        (font_large, "  FRESH MART GROCERY"),
        (font_small, "  123 Market Street"),
        (font_small, "  New York, NY 10001"),
        (font_small, "  Tel: (555) 123-4567"),
        (font_small, ""),
        (font_small, "  " + "-" * 32),
        (font_medium, "  Date: 03/15/2024"),
        (font_small, "  " + "-" * 32),
        (font_small, ""),
        (font_medium, "  Organic Milk      $4.99"),
        (font_medium, "  Wheat Bread       $3.49"),
        (font_medium, "  Fresh Eggs        $5.99"),
        (font_medium, "  Orange Juice      $4.29"),
        (font_medium, "  Cheddar Cheese    $6.99"),
        (font_medium, "  Bananas           $1.29"),
        (font_medium, "  Greek Yogurt      $3.79"),
        (font_small, ""),
        (font_small, "  " + "-" * 32),
        (font_medium, "  Subtotal:        $30.83"),
        (font_medium, "  Tax (8%):         $2.47"),
        (font_small, "  " + "-" * 32),
        (font_large, "  Total:           $33.30"),
        (font_small, "  " + "-" * 32),
        (font_small, ""),
        (font_medium, "  Payment: VISA **4521"),
        (font_small, ""),
        (font_small, "  Thank you for shopping!"),
        (font_small, "  Have a great day!"),
    ]

    for font, line in lines:
        if line == "":
            y += 10
            continue
        draw.text((10, y), line, fill="black", font=font)
        y += 25

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as JPEG
    img.save(output_path, "JPEG", quality=95)
    print(f"✅ Sample receipt saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_receipt()
