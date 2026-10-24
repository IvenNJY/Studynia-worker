import io
import json
import logging
import os
import time
import uuid
import zipfile
from html.parser import HTMLParser
from typing import List, Optional

import fitz
import pytesseract
import requests
from docx import Document
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from google import genai
from google.genai import types

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("studynia.worker")


def log_timing(event: str, started_at: float, **fields):
    """Write a structured duration record to stdout for Render logs."""
    record = {
        "event": event,
        "duration_seconds": round(time.perf_counter() - started_at, 3),
        **fields,
    }
    logger.info("timing %s", json.dumps(record, ensure_ascii=False, default=str))


def log_event(event: str, **fields):
    """Write a structured event record without exposing request credentials."""
    record = {"event": event, **fields}
    logger.info("event %s", json.dumps(record, ensure_ascii=False, default=str))


# Configure Gemini AI using standard environment variables.
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ Error: GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp"}
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
generation_started_at: dict[str, float] = {}

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


def ocr_image_bytes(image_bytes: bytes) -> str:
    """Extract readable text from one raster image."""
    started_at = time.perf_counter()
    image_width = None
    image_height = None
    try:
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            image_width, image_height = source_image.size
            image = source_image.convert("RGB")
        try:
            extracted_text = pytesseract.image_to_string(image).strip()
        finally:
            image.close()
    except Exception:
        log_timing(
            "image_ocr_failed",
            started_at,
            image_bytes=len(image_bytes),
            image_width=image_width,
            image_height=image_height,
        )
        raise

    log_timing(
        "image_ocr_completed",
        started_at,
        image_bytes=len(image_bytes),
        image_width=image_width,
        image_height=image_height,
        extracted_characters=len(extracted_text),
    )
    return extracted_text


def extract_pdf(file_bytes: bytes) -> str:
    """Extract a PDF's text layer and OCR its embedded images."""
    document = fitz.open(stream=file_bytes, filetype="pdf")
    extracted_pages = []
    for page_number, page in enumerate(document, start=1):
        page_parts = []
        native_text = page.get_text("text").strip()
        if native_text:
            page_parts.append(native_text)

        image_text_found = False
        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            image_xref = image_info[0]
            try:
                image_data = document.extract_image(image_xref)
                image_text = ocr_image_bytes(image_data["image"])
                if image_text:
                    image_text_found = True
                    page_parts.append(f"[Text detected in image {image_index}]\n{image_text}")
            except Exception as image_error:
                logger.warning(
                    "Could not read PDF image %s on page %s: %s",
                    image_index,
                    page_number,
                    image_error,
                )

        # Scanned pages sometimes expose no separate images in the PDF structure.
        # OCR the rendered page only when it has no extractable text or image text.
        if not native_text and not image_text_found:
            try:
                rendered_page = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                page_ocr = ocr_image_bytes(rendered_page.tobytes("png"))
                if page_ocr:
                    page_parts.append(f"[Text detected by page OCR]\n{page_ocr}")
            except Exception as render_error:
                logger.warning("Could not OCR PDF page %s: %s", page_number, render_error)

        if page_parts:
            extracted_pages.append(f"--- Page {page_number} ---\n" + "\n".join(page_parts))

    document.close()
    return "\n".join(extracted_pages)


def extract_pptx(file_bytes: bytes) -> str:
    """Extract slide text, table text, and OCR from embedded pictures."""
    presentation = Presentation(io.BytesIO(file_bytes))
    extracted_slides = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        slide_parts = []

        def extract_shapes(shapes, image_number=0):
            for shape in shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    image_number = extract_shapes(shape.shapes, image_number)
                    continue

                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_number += 1
                    try:
                        image_text = ocr_image_bytes(shape.image.blob)
                        if image_text:
                            slide_parts.append(f"[Text detected in image {image_number}]\n{image_text}")
                    except Exception as image_error:
                        # Linked pictures may not have image bytes embedded in the PPTX.
                        logger.warning(
                            "Could not read image %s on slide %s: %s",
                            image_number,
                            slide_number,
                            image_error,
                        )
                    continue

                if hasattr(shape, "text"):
                    shape_text = shape.text.strip()
                    if shape_text:
                        slide_parts.append(shape_text)

                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        row_text = "\t".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            slide_parts.append(row_text)

            return image_number

        extract_shapes(slide.shapes)
        if slide_parts:
            extracted_slides.append(f"--- Slide {slide_number} ---\n" + "\n".join(slide_parts))

    return "\n".join(extracted_slides)


def extract_docx(file_bytes: bytes) -> str:
    """Extract Word text and OCR images stored in a DOCX package."""
    document = Document(io.BytesIO(file_bytes))
    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]

    for table in document.tables:
        for row in table.rows:
            row_text = "\t".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)

    image_number = 0
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as package:
            image_names = [name for name in package.namelist() if name.startswith("word/media/")]
            for image_name in image_names:
                image_number += 1
                try:
                    image_text = ocr_image_bytes(package.read(image_name))
                    if image_text:
                        parts.append(f"[Text detected in image {image_number}]\n{image_text}")
                except Exception as image_error:
                    logger.warning("Could not read DOCX image %s: %s", image_name, image_error)
    except zipfile.BadZipFile as package_error:
        raise ValueError("The DOCX package is invalid") from package_error

    return "\n".join(parts)


def extract_text_in_memory(file_bytes: bytes, filename: str, content_type: str = "") -> str:
    """Read supported course resources, extracting native text and OCR where available."""
    normalized_filename = filename.lower().split("?")[0]
    extension = normalized_filename.rsplit(".", 1)[-1] if "." in normalized_filename else ""
    content_type = content_type.lower()

    if extension == "pdf" or "application/pdf" in content_type:
        return extract_pdf(file_bytes)

    if extension == "pptx" or "presentationml" in content_type:
        return extract_pptx(file_bytes)

    if extension == "docx" or "wordprocessingml" in content_type:
        return extract_docx(file_bytes)

    if extension in IMAGE_EXTENSIONS or content_type.startswith("image/"):
        return ocr_image_bytes(file_bytes)

    if extension in {"txt", "md", "csv", "html", "htm"} or content_type.startswith("text/"):
        decoded = file_bytes.decode("utf-8-sig", errors="replace")
        if extension in {"html", "htm"} or "text/html" in content_type:
            parser = TextOnlyHTMLParser()
            parser.feed(decoded)
            return "\n".join(part.strip() for part in parser.parts if part.strip())
        return decoded

    # Some LMS downloads use a generic content type despite containing an image.
    try:
        return ocr_image_bytes(file_bytes)
    except Exception:
        return ""

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
    supabase_url = (
        os.environ.get("SUPABASE_URL")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        or ""
    ).rstrip("/")
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
    persistence_started_at = time.perf_counter()
    job["persistence_status"] = "saving"
    generation_jobs[job_id] = job
    for attempt in range(3):
        attempt_started_at = time.perf_counter()
        try:
            persist_assessment_job(request, job)
            job["persistence_status"] = "saved"
            log_timing(
                "database_save_attempt_completed",
                attempt_started_at,
                job_id=job_id,
                attempt=attempt + 1,
                status="saved",
            )
            break
        except Exception as error:
            log_timing(
                "database_save_attempt_failed",
                attempt_started_at,
                job_id=job_id,
                attempt=attempt + 1,
                error_type=type(error).__name__,
            )
            job["persistence_error"] = str(error)
            if attempt < 2:
                time.sleep(2 ** attempt)
    else:
        job["persistence_status"] = "failed"

    generation_jobs[job_id] = job
    log_timing(
        "database_persistence_completed",
        persistence_started_at,
        job_id=job_id,
        status=job["persistence_status"],
    )

    # Support in-flight jobs created by the older callback-based app version.
    if request.callback_url and request.callback_token:
        send_generation_callback(job_id, request, job)

    job_started_at = generation_started_at.pop(job_id, time.perf_counter())
    log_timing(
        "generation_job_completed",
        job_started_at,
        job_id=job_id,
        status=job["status"],
        persistence_status=job["persistence_status"],
        file_count=len(request.files),
    )


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
        attempt_started_at = time.perf_counter()
        try:
            response = requests.post(
                request.callback_url,
                json=callback_payload,
                headers={"X-Assessment-Callback-Token": request.callback_token},
                timeout=30,
            )
            response.raise_for_status()
            log_timing(
                "callback_attempt_completed",
                attempt_started_at,
                job_id=job_id,
                attempt=attempt + 1,
            )
            return
        except Exception as error:
            log_timing(
                "callback_attempt_failed",
                attempt_started_at,
                job_id=job_id,
                attempt=attempt + 1,
                error_type=type(error).__name__,
            )
            if attempt < 2:
                time.sleep(2 ** attempt)


def generate_quiz_job(job_id: str, owner_id: str, request: CanvasQuizRequest):
    generation_started_at[job_id] = time.perf_counter()
    generation_jobs[job_id] = {"owner_id": owner_id, "status": "processing"}
    source_list = []
    combined_text = ""
    log_event(
        "generation_job_started",
        job_id=job_id,
        file_count=len(request.files),
    )

    for source_id, file_item in enumerate(request.files, start=1):
        source_list.append({"source_id": source_id, "filename": file_item.filename})
        file_started_at = time.perf_counter()
        stage = "download"

        try:
            parsed_url = requests.utils.urlparse(file_item.url)
            if parsed_url.scheme not in {"http", "https"}:
                raise ValueError("Only HTTP and HTTPS material links are supported")
            headers = {"Authorization": f"Bearer {request.canvas_token}"} if request.canvas_token else {}
            download_started_at = time.perf_counter()
            response = requests.get(file_item.url, headers=headers, timeout=(10, 90), stream=True)

            if response.status_code != 200:
                raise ValueError(f"Could not retrieve {file_item.filename} (HTTP {response.status_code})")

            max_bytes = 25 * 1024 * 1024
            file_bytes = bytearray()
            for chunk in response.iter_content(64 * 1024):
                file_bytes.extend(chunk)
                if len(file_bytes) > max_bytes:
                    raise ValueError(f"{file_item.filename} is larger than the 25 MB limit")

            log_timing(
                "file_download_completed",
                download_started_at,
                job_id=job_id,
                source_id=source_id,
                filename=file_item.filename,
                size_bytes=len(file_bytes),
            )

            stage = "extraction"
            extraction_started_at = time.perf_counter()
            extracted = extract_text_in_memory(
                bytes(file_bytes),
                file_item.filename,
                response.headers.get("content-type", ""),
            )
            log_timing(
                "file_extraction_completed",
                extraction_started_at,
                job_id=job_id,
                source_id=source_id,
                filename=file_item.filename,
                extracted_characters=len(extracted),
            )
            if not extracted.strip():
                extension = file_item.filename.lower().split("?")[0].rsplit(".", 1)[-1]
                if extension in {"pdf", "pptx", "docx", *IMAGE_EXTENSIONS}:
                    raise ValueError(
                        f"No readable text or image text could be extracted from {file_item.filename}. "
                        "It may contain only unsupported graphics or linked images; embed the images "
                        "or export the file to PDF and retry."
                    )
                raise ValueError(f"No readable text found in {file_item.filename}")
            combined_text += (
                f"\n[BEGIN SOURCE {source_id}: {file_item.filename}]\n"
                f"{extracted}\n[END SOURCE {source_id}]\n"
            )
            log_timing(
                "file_processing_completed",
                file_started_at,
                job_id=job_id,
                source_id=source_id,
                filename=file_item.filename,
                extracted_characters=len(extracted),
            )

        except Exception as e:
            log_timing(
                "file_processing_failed",
                file_started_at,
                job_id=job_id,
                source_id=source_id,
                filename=file_item.filename,
                stage=stage,
                error_type=type(e).__name__,
            )
            job = {
                "owner_id": owner_id,
                "status": "failed",
                "error": f"Failed processing {file_item.filename}: {str(e)}",
            }
            generation_jobs[job_id] = job
            finalize_generation_job(job_id, request, job)
            return

    if not combined_text.strip():
        job = {"owner_id": owner_id, "status": "failed", "error": "No readable text extracted from files."}
        generation_jobs[job_id] = job
        finalize_generation_job(job_id, request, job)
        return

    prompt = f"""
    Generate a multiple-choice quiz with {request.num_questions} questions based exclusively on the
    provided sources.

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

    ai_started_at = time.perf_counter()
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
        log_timing(
            "gemini_generation_completed",
            ai_started_at,
            job_id=job_id,
            question_count=len(quiz),
            source_count=len(source_list),
        )
        job = {
            "owner_id": owner_id,
            "status": "completed",
            "result": {"source_list": source_list, "quiz": quiz},
        }
        generation_jobs[job_id] = job
    except Exception as e:
        log_timing(
            "gemini_generation_failed",
            ai_started_at,
            job_id=job_id,
            error_type=type(e).__name__,
        )
        job = {
            "owner_id": owner_id,
            "status": "failed",
            "error": f"AI generation failed: {str(e)}",
        }
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
    database_url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    if not database_url or not os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
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
