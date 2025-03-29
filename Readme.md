# MeDecode AI Backend

This is an AI agent for medical report decoding built using FastAPI, NLP (SciSpaCy), and summarization (Transformers). The backend extracts text from PDFs using OCR, summarizes the extracted text to filter out extraneous information, detects medical symptoms and test results, and generates actionable health recommendations.

## Features
- **OCR Extraction:** Converts each PDF page to an image and extracts text using Tesseract OCR.
- **Summarization:** Uses a transformer model ("facebook/bart-large-cnn") to create a concise summary.
- **Medical Term Detection:** Leverages SciSpaCy to detect medical symptoms and map them to recommended specialists.
- **Test Result Interpretation:** Extracts test results and interprets them against predefined normal ranges.
- **Health Recommendations:** Generates recommendations for precautions, diet, and general advice based on the report.

## Requirements
- Python 3.8+
- [Poppler](https://poppler.freedesktop.org/) (for pdf2image)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## Installation
1. Clone the repository:
   ```bash
    git clone https://github.com/yourusername/medeecode-ai-backend.git
    cd medeecode-ai-backend

  Create and activate a virtual environment:

``` python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```
Install the dependencies:

   ``` pip install -r requirements.txt``` 

 Download and install Poppler and Tesseract as needed.

Running Locally

Run the app using:

```
uvicorn main:app --reload
```
Visit http://127.0.0.1:8000/docs to see the interactive API documentation.

Preview 
![image](https://github.com/user-attachments/assets/9474a3fd-0401-4052-b2f3-7ab731467ec5)

