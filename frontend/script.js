/* ============================================
   ReceiptAI — Frontend JavaScript
   Handles file upload, API calls, and results rendering
   ============================================ */

// ============================================
// API Configuration
// ============================================
// Auto-detect API base URL (same origin when served by FastAPI)
const API_BASE = window.location.origin;

// ============================================
// DOM Elements
// ============================================
const uploadSection = document.getElementById('upload-section');
const processingSection = document.getElementById('processing-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');

const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const removeFileBtn = document.getElementById('remove-file');
const extractBtn = document.getElementById('extract-btn');

const confidenceArc = document.getElementById('confidence-arc');
const confidenceValue = document.getElementById('confidence-value');
const vendorValue = document.getElementById('vendor-value');
const dateValue = document.getElementById('date-value');
const totalValue = document.getElementById('total-value');
const ocrTextBody = document.getElementById('ocr-text-body');

const downloadExcel = document.getElementById('download-excel');
const downloadCsv = document.getElementById('download-csv');
const downloadJson = document.getElementById('download-json');
const newUploadBtn = document.getElementById('new-upload-btn');

const errorMessage = document.getElementById('error-message');
const errorRetryBtn = document.getElementById('error-retry-btn');

// ============================================
// State
// ============================================
let selectedFile = null;
let currentFilename = null;  // Name used for download endpoints

// ============================================
// Section Navigation
// ============================================
function showSection(section) {
    // Hide all sections
    [uploadSection, processingSection, resultsSection, errorSection].forEach(s => {
        s.classList.remove('active');
        s.classList.add('hidden');
    });
    // Show the target section
    section.classList.remove('hidden');
    section.classList.add('active');
}

// ============================================
// File Size Formatting
// ============================================
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============================================
// Upload Zone — Drag & Drop Events
// ============================================
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

// ============================================
// File Selection Handler
// ============================================
function handleFileSelection(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
    const allowedExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(ext)) {
        showError('Unsupported file type. Please upload a JPG, PNG, BMP, TIFF, or WebP image.');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File is too large. Maximum size is 10MB.');
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    // Hide the upload zone animated text, show file preview
    uploadZone.style.display = 'none';
    filePreview.classList.remove('hidden');
}

// ============================================
// Remove File
// ============================================
removeFileBtn.addEventListener('click', () => {
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    currentFilename = null;
    fileInput.value = '';
    filePreview.classList.add('hidden');
    uploadZone.style.display = '';
    showSection(uploadSection);
}

// ============================================
// Extract Button — API Call
// ============================================
extractBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show processing screen
    showSection(processingSection);

    try {
        // Build form data with the file
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Call the full extraction endpoint
        const response = await fetch(`${API_BASE}/extract/full`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        currentFilename = data.filename;

        // Render the results
        renderResults(data);
        showSection(resultsSection);

    } catch (err) {
        showError(err.message || 'Failed to process the receipt. Please try again.');
    }
});

// ============================================
// Render Results
// ============================================
function renderResults(data) {
    const fields = data.fields;
    const words = data.words || [];
    const avgConf = data.average_confidence || 0;

    // --- Extracted Fields ---
    setFieldValue(vendorValue, fields.vendor);
    setFieldValue(dateValue, fields.date);
    setFieldValue(totalValue, fields.total ? `$${fields.total}` : null);

    // --- Confidence Ring ---
    animateConfidence(avgConf);

    // --- OCR Text with Highlighting ---
    renderHighlightedText(words, data.raw_text);
}

function setFieldValue(element, value) {
    if (value !== null && value !== undefined) {
        element.textContent = value;
        element.classList.remove('null-value');
    } else {
        element.textContent = 'Not detected';
        element.classList.add('null-value');
    }
}

// ============================================
// Confidence Ring Animation
// ============================================
function animateConfidence(percentage) {
    // Set the ring color based on confidence level
    let strokeColor;
    if (percentage >= 80) strokeColor = 'var(--conf-high)';
    else if (percentage >= 50) strokeColor = 'var(--conf-medium)';
    else strokeColor = 'var(--conf-low)';

    confidenceArc.style.stroke = strokeColor;

    // Animate from 0 to the actual value
    let current = 0;
    const target = Math.round(percentage);
    const duration = 1000;
    const startTime = performance.now();

    function animate(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        current = Math.round(eased * target);

        confidenceArc.setAttribute('stroke-dasharray', `${current}, 100`);
        confidenceValue.textContent = `${current}%`;

        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }

    requestAnimationFrame(animate);
}

// ============================================
// Render Highlighted OCR Text
// ============================================
function renderHighlightedText(words, rawText) {
    ocrTextBody.innerHTML = '';

    if (!words || words.length === 0) {
        ocrTextBody.textContent = rawText || 'No text detected.';
        return;
    }

    // Render each word as a span with confidence-based coloring
    // Track position in raw text to preserve line breaks
    const rawLines = (rawText || '').split('\n');
    let wordIndex = 0;

    rawLines.forEach((line, lineIdx) => {
        const lineWords = line.trim().split(/\s+/).filter(w => w);

        lineWords.forEach((rawWord) => {
            // Find matching word from confidence data
            let confidence = -1;
            if (wordIndex < words.length) {
                // Match by checking if words align
                const confWord = words[wordIndex];
                if (confWord && rawWord.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()
                    .includes(confWord.text.replace(/[^a-zA-Z0-9]/g, '').toLowerCase().substring(0, 3))) {
                    confidence = confWord.confidence;
                    wordIndex++;
                } else {
                    // Try a fuzzy match with next few words
                    for (let k = wordIndex; k < Math.min(wordIndex + 3, words.length); k++) {
                        if (words[k].text.toLowerCase().includes(rawWord.substring(0, 3).toLowerCase()) ||
                            rawWord.toLowerCase().includes(words[k].text.substring(0, 3).toLowerCase())) {
                            confidence = words[k].confidence;
                            wordIndex = k + 1;
                            break;
                        }
                    }
                }
            }

            // Determine confidence class
            let confClass = 'conf-medium';
            if (confidence >= 80) confClass = 'conf-high';
            else if (confidence >= 50) confClass = 'conf-medium';
            else if (confidence >= 0) confClass = 'conf-low';

            const span = document.createElement('span');
            span.className = `ocr-word ${confClass}`;
            span.textContent = rawWord;
            span.setAttribute('data-confidence', confidence >= 0 ? confidence : '?');
            span.title = `Confidence: ${confidence >= 0 ? confidence + '%' : 'unknown'}`;

            ocrTextBody.appendChild(span);
            ocrTextBody.appendChild(document.createTextNode(' '));
        });

        // Add line break between lines
        if (lineIdx < rawLines.length - 1) {
            ocrTextBody.appendChild(document.createElement('br'));
        }
    });
}

// ============================================
// Download Buttons
// ============================================
downloadExcel.addEventListener('click', () => downloadFile('excel'));
downloadCsv.addEventListener('click', () => downloadFile('csv'));
downloadJson.addEventListener('click', () => downloadFile('json'));

function downloadFile(format) {
    if (!currentFilename) return;

    // Trigger browser download
    const url = `${API_BASE}/download/${encodeURIComponent(currentFilename)}?format=${format}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// ============================================
// New Upload Button
// ============================================
newUploadBtn.addEventListener('click', () => {
    resetUpload();
});

// ============================================
// Error Handling
// ============================================
function showError(message) {
    errorMessage.textContent = message;
    showSection(errorSection);
}

errorRetryBtn.addEventListener('click', () => {
    resetUpload();
});
