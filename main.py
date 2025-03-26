from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF
import spacy
import os
from fastapi.middleware.cors import CORSMiddleware
import re
from transformers import pipeline, AutoTokenizer

# Initialize summarizer pipeline with a lighter model for memory efficiency
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (frontend domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load SciSpaCy NLP model with large medical model
nlp = spacy.load("en_core_sci_lg")

# Specialist mapping based on detected symptoms
SPECIALIST_MAPPING = {
    "chest pain": "Cardiologist",
    "heart attack": "Cardiologist",
    "diabetes": "Endocrinologist",
    "thyroid": "Endocrinologist",
    "stomach pain": "Gastroenterologist",
    "liver": "Gastroenterologist",
    "cough": "Pulmonologist",
    "breathing problem": "Pulmonologist",
    "skin rash": "Dermatologist",
    "headache": "Neurologist",
    "seizures": "Neurologist",
    "kidney": "Nephrologist",
    "arthritis": "Rheumatologist"
}

# Medical test normal ranges and explanations
MEDICAL_TESTS = {
    "hemoglobin": {
        "normal_range": "13.8-17.2 g/dL (male), 12.1-15.1 g/dL (female)",
        "meaning": "Hemoglobin carries oxygen in your blood. Low levels may indicate anemia."
    },
    "cholesterol": {
        "normal_range": "Less than 200 mg/dL",
        "meaning": "High cholesterol can increase the risk of heart disease."
    },
    "blood glucose": {
        "normal_range": "70-99 mg/dL (fasting)",
        "meaning": "Elevated blood sugar levels can indicate diabetes."
    }
}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF by converting each page to an image and using Tesseract OCR.
    """
    images = convert_from_path(pdf_path, dpi=300)  # Higher DPI for better OCR accuracy
    text = ""
    for img in images:
        extracted_text = pytesseract.image_to_string(img, lang="eng")
        text += extracted_text + "\n"
    return text.strip()


def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image_path)


def summarize_extracted_text(text: str) -> str:
    """
    Summarize the extracted text using a transformer model.
    Splits the text into smaller chunks based on token count and summarizes each.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    total_tokens = inputs.input_ids.size(1)
    max_tokens = 500  # Maximum tokens per chunk

    if total_tokens <= max_tokens:
        try:
            summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
            return summary[0]['summary_text']
        except Exception:
            return text  # Fallback to original text if summarization fails

    # Otherwise, split text into chunks based on words (approximate token count)
    words = text.split()
    chunk_size = max_tokens  # Rough approximation: 1 token ~ 1 word
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    summaries = []
    for chunk in chunks:
        try:
            summ = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
            summaries.append(summ[0]['summary_text'])
        except Exception:
            summaries.append(chunk)  # Fallback to chunk if error occurs

    combined_summary = " ".join(summaries)
    combined_tokens = tokenizer(combined_summary, return_tensors="pt", truncation=False).input_ids.size(1)
    if combined_tokens > max_tokens:
        try:
            final_summary = summarizer(combined_summary, max_length=200, min_length=50, do_sample=False)
            return final_summary[0]['summary_text']
        except Exception:
            return combined_summary
    else:
        return combined_summary


def detect_medical_terms(text: str):
    """
    Detect medical symptoms and test results.
    Returns:
      detected_symptoms: dict mapping symptom to specialist,
      recommended_specialists: list,
      test_results: dict of test name to extracted value and details.
    """
    doc = nlp(text)
    detected_terms = {ent.text.lower(): ent.label_ for ent in doc.ents}
    detected_symptoms = {term: SPECIALIST_MAPPING[term] for term in detected_terms if term in SPECIALIST_MAPPING}
    recommended_specialists = list(set(detected_symptoms.values())) if detected_symptoms else ["General Physician"]

    test_results = {}
    for test_name in MEDICAL_TESTS:
        if test_name in text.lower():
            match = re.search(rf"{test_name}[:\s]+(\d+\.?\d*)", text, re.IGNORECASE)
            if match:
                test_value = match.group(1)
                test_results[test_name] = {
                    "value": test_value,
                    "normal_range": MEDICAL_TESTS[test_name]["normal_range"],
                    "meaning": MEDICAL_TESTS[test_name]["meaning"]
                }
    return detected_symptoms, recommended_specialists, test_results


def generate_recommendations(detected_symptoms, test_results):
    """
    Generate health recommendations based on detected symptoms and test results.
    Returns a dictionary with recommendations for precautions, diet, medications, and general advice.
    """
    recommendations = {
        "precautions": [],
        "diet": [],
        "medications": [],
        "general_advice": []
    }

    # Recommendations based on detected symptoms
    for symptom in detected_symptoms.keys():
        if symptom == "diabetes":
            recommendations["precautions"].append("Monitor blood sugar levels regularly.")
            recommendations["diet"].append("Follow a low-sugar, high-fiber diet.")
            recommendations["general_advice"].append("Consult an endocrinologist for diabetes management.")
        elif symptom in ["chest pain", "heart attack"]:
            recommendations["precautions"].append("Avoid strenuous activities until evaluated.")
            recommendations["diet"].append("Adopt a heart-healthy diet low in saturated fats.")
            recommendations["general_advice"].append("Consult a cardiologist immediately.")
        elif symptom == "liver":
            recommendations["precautions"].append("Avoid alcohol and hepatotoxic substances.")
            recommendations["diet"].append("Increase intake of antioxidants (fruits and vegetables).")
            recommendations["general_advice"].append("Consult a gastroenterologist for liver evaluation.")
        elif symptom == "thyroid":
            recommendations["precautions"].append("Monitor thyroid levels periodically.")
            recommendations["diet"].append("Include iodine-rich foods in your diet.")
            recommendations["general_advice"].append("Consult an endocrinologist for thyroid management.")
        elif symptom in ["cough", "breathing problem"]:
            recommendations["precautions"].append("Avoid smoking and exposure to pollutants.")
            recommendations["general_advice"].append("Consult a pulmonologist if symptoms persist.")

    # Recommendations based on test results
    for test, details in test_results.items():
        try:
            value = float(details["value"])
        except ValueError:
            continue

        if test == "cholesterol" and value >= 200:
            recommendations["diet"].append("Reduce intake of saturated fats and increase dietary fiber.")
            recommendations["general_advice"].append("Consult a nutritionist and cardiologist for cholesterol management.")
        if test == "blood glucose" and value >= 100:
            recommendations["diet"].append("Reduce simple carbohydrates and sugars.")
            recommendations["general_advice"].append("Monitor blood glucose levels and consult an endocrinologist.")

    for key in recommendations:
        recommendations[key] = list(set(recommendations[key]))
    return recommendations


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Handle file upload, extract text using OCR, generate a summary,
    detect medical terms and test results, and generate health recommendations.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    extracted_text = (
        extract_text_from_pdf(file_path)
        if file.filename.endswith(".pdf")
        else extract_text_from_image(file_path)
    )
    summarized_text = summarize_extracted_text(extracted_text)
    detected_symptoms, recommended_specialists, test_results = detect_medical_terms(summarized_text)
    recommendations = generate_recommendations(detected_symptoms, test_results)

    return {
        "filename": file.filename,
        "extracted_text": summarized_text,
        "detected_symptoms": detected_symptoms,
        "recommended_specialists": recommended_specialists,
        "test_results": test_results,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

