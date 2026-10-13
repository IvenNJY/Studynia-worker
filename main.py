import os
import io
import json
import requests
import uuid
from html.parser import HTMLParser
from typing import List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException
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

# Jobs stay in the worker process while a browser polls their status. The
# generation itself runs after the request returns, so switching tabs does not
# cancel it.
generation_jobs: dict[str, dict] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",],
    # allow * for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextOnlyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def extract_text_in_memory(file_bytes: bytes, filename: str, content_type: str = "") -> str:
    """Read supported course resources directly from memory."""
    text = ""
    extension = filename.lower().split("?")[0].rsplit(".", 1)[-1] if "." in filename else ""
    content_type = content_type.lower()
    if extension == "pdf" or "application/pdf" in content_type:
        images = convert_from_bytes(file_bytes)
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            text += f"--- Page {i+1} ---\n{page_text}\n"

    elif extension == "pptx" or "presentationml" in content_type:
        prs = Presentation(io.BytesIO(file_bytes))
        for i, slide in enumerate(prs.slides):
            slide_text = "\n".join([shape.text for shape in slide.shapes if hasattr(shape, "text")])
            text += f"--- Slide {i+1} ---\n{slide_text}\n"
    elif extension in {"txt", "md", "csv", "html", "htm"} or content_type.startswith("text/"):
        decoded = file_bytes.decode("utf-8-sig", errors="replace")
        if extension in {"html", "htm"} or "text/html" in content_type:
            parser = TextOnlyHTMLParser()
            parser.feed(decoded)
            text = "\n".join(part.strip() for part in parser.parts if part.strip())
        else:
            text = decoded

    return text

class FileItem(BaseModel):
    url: str
    filename: str  

class CanvasQuizRequest(BaseModel):
    files: List[FileItem]
    owner_id: str
    canvas_token: Optional[str] = None
    num_questions: int = 5

def generate_quiz_job(job_id: str, owner_id: str, request: CanvasQuizRequest):
    generation_jobs[job_id] = {"owner_id": owner_id, "status": "processing"}
    source_list = []
    combined_text = ""

    for source_id, file_item in enumerate(request.files, start=1):
        source_list.append({"source_id": source_id, "filename": file_item.filename})

        try:
            parsed_url = requests.utils.urlparse(file_item.url)
            if parsed_url.scheme not in {"http", "https"}:
                raise ValueError("Only HTTP and HTTPS material links are supported")
            headers = {"Authorization": f"Bearer {request.canvas_token}"} if request.canvas_token else {}
            response = requests.get(file_item.url, headers=headers, timeout=(10, 90), stream=True)

            if response.status_code != 200:
                raise ValueError(f"Could not retrieve {file_item.filename} (HTTP {response.status_code})")

            max_bytes = 25 * 1024 * 1024
            file_bytes = bytearray()
            for chunk in response.iter_content(64 * 1024):
                file_bytes.extend(chunk)
                if len(file_bytes) > max_bytes:
                    raise ValueError(f"{file_item.filename} is larger than the 25 MB limit")
            extracted = extract_text_in_memory(bytes(file_bytes), file_item.filename, response.headers.get("content-type", ""))
            if not extracted.strip():
                raise ValueError(f"No readable text found in {file_item.filename}")
            combined_text += f"\n[BEGIN SOURCE {source_id}: {file_item.filename}]\n{extracted}\n[END SOURCE {source_id}]\n"

        except Exception as e:
            generation_jobs[job_id] = {"owner_id": owner_id, "status": "failed", "error": f"Failed processing {file_item.filename}: {str(e)}"}
            return

    if not combined_text.strip():
        generation_jobs[job_id] = {"owner_id": owner_id, "status": "failed", "error": "No readable text extracted from files."}
        return

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
        quiz = quiz_data.get("quiz", [])
        if not isinstance(quiz, list) or not quiz:
            raise ValueError("The AI returned no quiz questions")
        generation_jobs[job_id] = {"owner_id": owner_id, "status": "completed", "result": {
            "source_list": source_list,
            "quiz": quiz
        }}
    except Exception as e:
        generation_jobs[job_id] = {"owner_id": owner_id, "status": "failed", "error": f"AI generation failed: {str(e)}"}


@app.post("/generate-quiz-from-urls")
async def generate_quiz_from_urls_endpoint(request: CanvasQuizRequest, background_tasks: BackgroundTasks):
    if not request.files:
        raise HTTPException(status_code=400, detail="At least one material link is required")
    if not request.owner_id:
        raise HTTPException(status_code=400, detail="An owner id is required")
    job_id = str(uuid.uuid4())
    generation_jobs[job_id] = {"owner_id": request.owner_id, "status": "queued"}
    background_tasks.add_task(generate_quiz_job, job_id, request.owner_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/generate-quiz-from-urls/{job_id}")
async def get_generation_job(job_id: str, owner_id: str):
    job = generation_jobs.get(job_id)
    if not job or job.get("owner_id") != owner_id:
        raise HTTPException(status_code=404, detail="Generation job not found")
    return {"job_id": job_id, **{key: value for key, value in job.items() if key != "owner_id"}}
