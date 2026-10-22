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
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from google import genai
from google.genai import types

# 1. Configure Gemini AI using standard environment variables
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("❌ Error: GEMINI_API_KEY environment variable not set.")
    
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = (
    "You are an academic quiz generator. You must base all questions and answers "
    "strictly on the provided source materials without inventing outside facts."
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
            slide_parts = [shape.text.strip() for shape in slide.shapes
                           if hasattr(shape, "text") and shape.text.strip()]

            def add_image_ocr(shapes, image_number):
                for shape in shapes:
                    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                        image_number = add_image_ocr(shape.shapes, image_number)
                    elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        image_number += 1
                        image = Image.open(io.BytesIO(shape.image.blob)).convert("RGB")
                        image_text = pytesseract.image_to_string(image).strip()
                        if image_text:
                            slide_parts.append(f"[Text detected in image {image_number}]\n{image_text}")
                return image_number

            add_image_ocr(slide.shapes, 0)
            slide_text = "\n".join(slide_parts)
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
    assessment_id: str
    job_id: str
    callback_url: Optional[str] = None
    callback_token: Optional[str] = None
    canvas_token: Optional[str] = None
    num_questions: int = 5


def supabase_request(method: str, table: str, *, params=None, payload=None):
    supabase_url = (os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or "").rstrip("/")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured on the worker")

    response = requests.request(
        method,
        f"{supabase_url}/rest/v1/{table}",
        params=params,
        json=payload,
        headers={
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        timeout=(10, 45),
    )
    response.raise_for_status()
    return response


def create_assessment_notification(request: CanvasQuizRequest, status: str, error: Optional[str] = None):
    existing = supabase_request(
        "GET",
        "inbox_notifications",
        params={
            "select": "id",
            "recipient_id": f"eq.{request.owner_id}",
            "assessment_id": f"eq.{request.assessment_id}",
            "notification_type": "eq.assessment_share",
            "limit": "1",
        },
    ).json()
    if existing:
        return

    ready = status == "completed"
    title = "Your assessment is ready" if ready else "Assessment generation failed"
    body = (
        f"Your assessment is ready in Past Assessments. Open Practice to start it."
        if ready
        else f"Your assessment could not be generated. {error or 'Please try again from Practice.'}"
    )
    supabase_request(
        "POST",
        "inbox_notifications",
        payload={
            "recipient_id": request.owner_id,
            "actor_id": request.owner_id,
            "notification_type": "assessment_share",
            "assessment_id": request.assessment_id,
            "title": title,
            "body": body,
        },
    )


def persist_assessment_job(request: CanvasQuizRequest, job: dict):
    if job["status"] == "completed":
        quiz = job.get("result", {}).get("quiz", [])
        if not quiz:
            raise ValueError("Generated assessment contains no questions")

        existing_questions = supabase_request(
            "GET",
            "assessment_questions",
            params={"select": "id", "assessment_id": f"eq.{request.assessment_id}", "limit": "1"},
        ).json()
        if not existing_questions:
            supabase_request(
                "POST",
                "assessment_questions",
                payload=[
                    {
                        "assessment_id": request.assessment_id,
                        "source_id": question.get("source_id"),
                        "question_text": question["question"],
                        "question_type": "mcq",
                        "options": question["options"],
                        "correct_answer": question["correct_answer"],
                        "explanation": question.get("explanation"),
                    }
                    for question in quiz
                ],
            )

        source = {
            "status": "ready",
            "job_id": request.job_id,
            "source_list": job.get("result", {}).get("source_list", []),
        }
    else:
        source = {
            "status": "failed",
            "job_id": request.job_id,
            "error": (job.get("error") or "Assessment generation failed")[:1000],
        }

    supabase_request(
        "PATCH",
        "assessments",
        params={"id": f"eq.{request.assessment_id}", "user_id": f"eq.{request.owner_id}"},
        payload={"source": json.dumps(source)},
    )
    create_assessment_notification(request, job["status"], job.get("error"))


def finalize_generation_job(job_id: str, request: CanvasQuizRequest, job: dict):
    job["persistence_status"] = "saving"
    generation_jobs[job_id] = job
    for attempt in range(3):
        try:
            persist_assessment_job(request, job)
            job["persistence_status"] = "saved"
            break
        except Exception as error:
            print(f"Assessment database save attempt {attempt + 1} failed: {error}")
            job["persistence_error"] = str(error)
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)
    else:
        job["persistence_status"] = "failed"

    generation_jobs[job_id] = job
    # Support in-flight jobs created by the older callback-based app version.
    if request.callback_url and request.callback_token:
        send_generation_callback(job_id, request, job)


def send_generation_callback(job_id: str, request: CanvasQuizRequest, job: dict):
    callback_payload = {
        "job_id": job_id,
        "owner_id": request.owner_id,
        "assessment_id": request.assessment_id,
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
    }
    for attempt in range(3):
        try:
            response = requests.post(
                request.callback_url,
                json=callback_payload,
                headers={"X-Assessment-Callback-Token": request.callback_token},
                timeout=30,
            )
            response.raise_for_status()
            return
        except Exception as error:
            print(f"Assessment callback attempt {attempt + 1} failed: {error}")
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)

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
            job = {"owner_id": owner_id, "status": "failed", "error": f"Failed processing {file_item.filename}: {str(e)}"}
            generation_jobs[job_id] = job
            finalize_generation_job(job_id, request, job)
            return

    if not combined_text.strip():
        job = {"owner_id": owner_id, "status": "failed", "error": "No readable text extracted from files."}
        generation_jobs[job_id] = job
        finalize_generation_job(job_id, request, job)
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
        ai_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                response_mime_type="application/json",
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )
        quiz_data = json.loads(ai_response.text)
        quiz = quiz_data.get("quiz", [])
        if not isinstance(quiz, list) or not quiz:
            raise ValueError("The AI returned no quiz questions")
        quiz = quiz[:request.num_questions]
        job = {"owner_id": owner_id, "status": "completed", "result": {
            "source_list": source_list,
            "quiz": quiz
        }}
        generation_jobs[job_id] = job
    except Exception as e:
        job = {"owner_id": owner_id, "status": "failed", "error": f"AI generation failed: {str(e)}"}
        generation_jobs[job_id] = job
    finalize_generation_job(job_id, request, job)


@app.post("/generate-quiz-from-urls")
async def generate_quiz_from_urls_endpoint(request: CanvasQuizRequest, background_tasks: BackgroundTasks):
    # Enforce the limit at the worker boundary as clients can bypass the web app API.
    request.num_questions = min(15, max(1, request.num_questions))
    if not request.files:
        raise HTTPException(status_code=400, detail="At least one material link is required")
    if not request.owner_id or not request.assessment_id:
        raise HTTPException(status_code=400, detail="Generation job metadata is required")
    if not (os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")) or not os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
        raise HTTPException(status_code=503, detail="Worker database persistence is not configured")
    job_id = request.job_id or str(uuid.uuid4())
    generation_jobs[job_id] = {"owner_id": request.owner_id, "status": "queued"}
    background_tasks.add_task(generate_quiz_job, job_id, request.owner_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/generate-quiz-from-urls/{job_id}")
async def get_generation_job(job_id: str, owner_id: str):
    job = generation_jobs.get(job_id)
    if not job or job.get("owner_id") != owner_id:
        raise HTTPException(status_code=404, detail="Generation job not found")
    return {"job_id": job_id, **{key: value for key, value in job.items() if key != "owner_id"}}
