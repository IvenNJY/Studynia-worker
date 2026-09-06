import os
import io
import json
import requests
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pytesseract
from pdf2image import convert_from_bytes
from pptx import Presentation
import google.generativeai as genai

# 1. Configure Gemini AI using standard environment variables
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("❌ Error: GEMINI_API_KEY environment variable not set.")
    
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    system_instruction=(
        "You are an academic quiz generator. You must base all questions and answers "
        "strictly on the provided source materials without inventing outside facts."
    ),
    generation_config={
        "temperature": 0.2,
        "top_p": 0.95,
        "response_mime_type": "application/json"
    }
)

# 2. Configure FastAPI Application
app = FastAPI(title="LMS Quiz Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_in_memory(file_bytes: bytes, filename: str) -> str:
    """Helper function to read text from PDFs or PPTXs directly from memory."""
    text = ""
    if filename.lower().endswith(".pdf"):
        images = convert_from_bytes(file_bytes)
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            text += f"--- Page {i+1} ---\n{page_text}\n"

    elif filename.lower().endswith(".pptx"):
        prs = Presentation(io.BytesIO(file_bytes))
        for i, slide in enumerate(prs.slides):
            slide_text = "\n".join([shape.text for shape in slide.shapes if hasattr(shape, "text")])
            text += f"--- Slide {i+1} ---\n{slide_text}\n"

    return text

class FileItem(BaseModel):
    url: str
    filename: str  

class CanvasQuizRequest(BaseModel):
    files: List[FileItem]
    canvas_token: str
    num_questions: int = 5

@app.post("/generate-quiz-from-urls")
async def generate_quiz_from_urls_endpoint(request: CanvasQuizRequest):
    source_list = []
    combined_text = ""

    headers = {
        "Authorization": f"Bearer {request.canvas_token}"
    }

    for source_id, file_item in enumerate(request.files, start=1):
        source_list.append({"source_id": source_id, "filename": file_item.filename})

        try:
            response = requests.get(file_item.url, headers=headers)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download {file_item.filename} from Canvas. Status Code: {response.status_code}"
                )

            file_bytes = response.content
            extracted = extract_text_in_memory(file_bytes, file_item.filename)
            combined_text += f"\n[BEGIN SOURCE {source_id}: {file_item.filename}]\n{extracted}\n[END SOURCE {source_id}]\n"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed processing {file_item.filename}: {str(e)}")

    if not combined_text.strip():
        raise HTTPException(status_code=400, detail="No readable text extracted from files.")

    prompt = f"""
    Generate a multiple-choice quiz with {request.num_questions} questions based exclusively on the provided sources.

    Strictly adhere to this JSON format:
    {{
      "quiz": [
        {{
          "question": "Question text here",
          "options": {{
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D"
          }},
          "correct_answer": "A",
          "explanation": "Brief explanation referencing the source material.",
          "source_id": 1
        }}
      ]
    }}

    Extracted Content:
    {combined_text}
    """

    try:
        ai_response = model.generate_content(prompt)
        quiz_data = json.loads(ai_response.text)
        return {
            "source_list": source_list,
            "quiz": quiz_data.get("quiz", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
