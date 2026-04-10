# 🧾 OCR Receipt Extraction Tool

A Python-based OCR tool that extracts structured data (vendor name, date, total amount) from receipt and invoice images. Returns results as JSON, CSV, and Excel — with a FastAPI REST API for integration with frontends.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 📷 **Image Preprocessing** — Grayscale, denoising, and Otsu thresholding via OpenCV
- 🔍 **OCR Text Extraction** — Powered by Tesseract OCR engine
- 🎯 **Smart Field Extraction** — Regex-based extraction of vendor, date, and total
- 📊 **Confidence Scores** — Word-level OCR confidence for quality assessment
- 💾 **Multi-format Export** — JSON, CSV, and Excel (.xlsx) output
- 🚀 **REST API** — FastAPI endpoints for frontend integration
- 🖥️ **CLI Tool** — Process receipts directly from the command line
- 🐳 **Docker Ready** — Containerized for easy deployment

---

## 📁 Project Structure

```
receipt-ocr/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with REST endpoints
│   ├── extractor.py         # Regex-based field extraction (vendor, date, total)
│   ├── preprocessor.py      # OpenCV image preprocessing pipeline
│   └── ocr.py               # Tesseract OCR wrapper with confidence scores
├── utils/
│   ├── __init__.py
│   └── file_handler.py      # JSON, CSV, and Excel export functions
├── samples/
│   └── sample_receipt.jpg   # Sample receipt image for testing
├── outputs/
│   └── .gitkeep             # Directory for saved results
├── tests/
│   ├── __init__.py
│   └── test_extractor.py    # Unit tests for extraction logic
├── run.py                   # CLI entry point
├── generate_sample.py       # Sample receipt image generator
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker container configuration
├── docker-compose.yml       # Docker Compose setup
└── README.md                # This file
```

---

## 🛠️ Installation

### 1. Install Tesseract OCR Engine

Tesseract must be installed on your system separately from the Python package.

**macOS:**
```bash
brew install tesseract
```

**Ubuntu / Debian Linux:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**Windows:**
1. Download the installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR`)
3. Add to your system PATH:
   - Open System Properties → Environment Variables
   - Add `C:\Program Files\Tesseract-OCR` to the `Path` variable
4. Verify: open Command Prompt and run `tesseract --version`

### 2. Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/ibrahimshaharyar/OCR-Project.git
cd OCR-Project

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Usage: CLI (Command Line)

Process a single receipt image from the terminal:

```bash
python run.py samples/sample_receipt.jpg
```

With a custom output filename:

```bash
python run.py samples/sample_receipt.jpg --output my_receipt
```

**What it does:**
1. Preprocesses the image (grayscale → denoise → threshold)
2. Runs Tesseract OCR to extract text
3. Parses vendor, date, and total using regex
4. Prints the result as JSON in the terminal
5. Saves JSON, CSV, and Excel files to `outputs/`

---

## 🚀 Usage: FastAPI (REST API)

Start the API server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/extract` | Extract fields from uploaded image |
| `POST` | `/extract/full` | Full extraction with raw text + confidence |
| `GET` | `/download/{filename}?format=json` | Download saved results |

### Example: Upload via cURL

**Basic extraction:**
```bash
curl -X POST -F "file=@samples/sample_receipt.jpg" http://localhost:8000/extract
```

**Full extraction with confidence data:**
```bash
curl -X POST -F "file=@samples/sample_receipt.jpg" http://localhost:8000/extract/full
```

**Download results as Excel:**
```bash
curl -O http://localhost:8000/download/sample_receipt?format=excel
```

### Interactive API Docs

Once the server is running, visit: **http://localhost:8000/docs**

---

## 📋 Sample Output

### JSON Output
```json
{
    "vendor": "FRESH MART GROCERY",
    "date": "03/15/2024",
    "total": "33.30",
    "confidence": 87.5
}
```

### CSV Output
```csv
vendor,date,total,confidence
FRESH MART GROCERY,03/15/2024,33.30,87.5
```

### Full API Response (`/extract/full`)
```json
{
    "fields": {
        "vendor": "FRESH MART GROCERY",
        "date": "03/15/2024",
        "total": "33.30"
    },
    "raw_text": "FRESH MART GROCERY\n123 Market Street\n...",
    "words": [
        {"text": "FRESH", "confidence": 95},
        {"text": "MART", "confidence": 92},
        {"text": "GROCERY", "confidence": 88}
    ],
    "average_confidence": 87.5,
    "filename": "sample_receipt"
}
```

---

## 🐳 Docker

### Build and run with Docker Compose:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### Build manually:

```bash
docker build -t receipt-ocr .
docker run -p 8000:8000 receipt-ocr
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

The unit tests cover:
- Vendor extraction (first meaningful line detection)
- Date extraction (all supported formats: DD/MM/YYYY, MM-DD-YYYY, YYYY-MM-DD, month names)
- Total extraction (keywords: total, grand total, amount due, balance due)
- Graceful handling of missing fields (returns `null`)
- Edge cases (empty input, special characters)

---

## 📅 Supported Date Formats

| Format | Example |
|--------|---------|
| DD/MM/YYYY | 15/01/2024 |
| MM-DD-YYYY | 01-15-2024 |
| YYYY-MM-DD | 2024-01-15 |
| Month DD, YYYY | January 15, 2024 |
| DD Month YYYY | 15 January 2024 |
| Mon DD, YYYY | Jan 15, 2024 |
| DD Mon YYYY | 15 Jan 2024 |

---

## 💰 Supported Total Keywords

The tool looks for these keywords (case-insensitive) followed by a numeric value:

- `Total`
- `Grand Total`
- `Amount Due`
- `Balance Due`
- `Total Due`
- `Amount`

---

## 📝 License

MIT License — free for personal and commercial use.
