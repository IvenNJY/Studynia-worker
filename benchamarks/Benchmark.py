import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import fitz
from docx import Document
from google import genai
from google.genai import types
from pptx import Presentation



INPUT_FILE = "/content/Wk1.pptx"
QUESTION_COUNT = 5
OUTPUT_FILE = "/content/gemini_benchmark_results.json"
OUTPUT_CSV_FILE = "/content/gemini_benchmark_results.csv"
MAX_RETRIES = 4
MODEL_DELAY_SECONDS = 5

MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]


SYSTEM_INSTRUCTION = (
    "You are an academic quiz generator. Base every question and answer "
    "strictly on the supplied source material."
)


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as document:
            return "\n".join(page.get_text("text") for page in document)

    if path.suffix.lower() == ".docx":
        document = Document(path)
        parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append("\t".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    if path.suffix.lower() == ".pptx":
        presentation = Presentation(path)
        parts = []
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text)
        return "\n".join(parts)

    raise ValueError("Supported file types are .txt, .pdf, .docx, and .pptx")


def parse_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "").strip()
    return json.loads(cleaned)


def quality_score(result: dict[str, Any], expected_count: int) -> float:
    quiz = result.get("quiz")
    if not isinstance(quiz, list) or not quiz:
        return 0.0

    question_scores = []
    for question in quiz:
        score = 0.0
        if not isinstance(question, dict):
            question_scores.append(score)
            continue

        if isinstance(question.get("question"), str) and question["question"].strip():
            score += 25

        options = question.get("options")
        if (
            isinstance(options, dict)
            and set(options) == {"A", "B", "C", "D"}
            and all(isinstance(value, str) and value.strip() for value in options.values())
        ):
            score += 25

        if question.get("correct_answer") in {"A", "B", "C", "D"}:
            score += 25

        if isinstance(question.get("explanation"), str) and question["explanation"].strip():
            score += 25

        question_scores.append(score)

    structure_score = sum(question_scores) / len(question_scores)
    count_score = min(len(quiz), expected_count) / expected_count * 100
    return round(structure_score * 0.75 + count_score * 0.25, 2)


def get_usage_value(usage: Any, attribute: str) -> int | None:
    value = getattr(usage, attribute, None)
    if value is not None:
        return int(value)
    if isinstance(usage, dict) and usage.get(attribute) is not None:
        return int(usage[attribute])
    return None


def generate_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
) -> tuple[Any, float]:
    """Return the response and API-call time, excluding retry waits."""
    api_seconds = 0.0

    for attempt in range(MAX_RETRIES):
        request_started_at = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.95,
                    response_mime_type="application/json",
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )
            api_seconds += time.perf_counter() - request_started_at
            return response, api_seconds
        except Exception as error:
            api_seconds += time.perf_counter() - request_started_at
            error_text = str(error).upper()
            temporarily_unavailable = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            )

            if not temporarily_unavailable or attempt == MAX_RETRIES - 1:
                raise

            wait_seconds = 2 ** attempt
            print(
                f"{model} is temporarily unavailable. "
                f"Retry {attempt + 1}/{MAX_RETRIES - 1} in "
                f"{wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)


def benchmark_model(
    client: genai.Client,
    model: str,
    source: str,
    question_count: int,
) -> dict[str, Any]:
    prompt = f"""
Generate {question_count} multiple-choice questions using only this source.

Return valid JSON in exactly this format:
{{
  "quiz": [
    {{
      "question": "Question text",
      "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
      "correct_answer": "A",
      "explanation": "Explanation based on the source"
    }}
  ]
}}

Source material:
{source}
"""

    api_seconds = 0.0
    try:
        response, api_seconds = generate_with_retry(client, model, prompt)
        parsed = parse_response(response.text or "")
        quiz = parsed.get("quiz", [])
        usage = getattr(response, "usage_metadata", None)
        return {
            "model": model,
            "status": "success",
            "latency_seconds": round(api_seconds, 3),
            "prompt_tokens": get_usage_value(usage, "prompt_token_count"),
            "output_tokens": get_usage_value(usage, "candidates_token_count"),
            "json_valid": True,
            "quality_score": quality_score(parsed, question_count),
            "question_count": len(quiz) if isinstance(quiz, list) else 0,
            "generated_quiz": parsed,
            "error": "",
        }
    except Exception as error:
        return {
            "model": model,
            "status": "failed",
            "latency_seconds": round(api_seconds, 3),
            "prompt_tokens": None,
            "output_tokens": None,
            "json_valid": False,
            "quality_score": 0.0,
            "question_count": 0,
            "generated_quiz": None,
            "error": f"{type(error).__name__}: {error}",
        }


def save_results(results: list[dict[str, Any]], output: Path, csv_output: Path) -> None:
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_fields = [
        "model",
        "status",
        "latency_seconds",
        "prompt_tokens",
        "output_tokens",
        "json_valid",
        "quality_score",
        "question_count",
        "error",
    ]
    with csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(
            {field: result.get(field) for field in csv_fields}
            for result in results
        )


def main() -> None:
    print("Gemini benchmark starting.")
    print(f"Input file: {INPUT_FILE}")
    print("If this is not your uploaded file, change INPUT_FILE in the script and run again.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY in Colab before running, for example: "
            'os.environ["GEMINI_API_KEY"] = "your-api-key"'
        )

    source_path = Path(INPUT_FILE)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Input file was not found: {source_path}. "
            "Upload the file to Colab or change INPUT_FILE in this script."
        )

    source = extract_text(source_path).strip()
    if not source:
        raise ValueError("No readable text was extracted from the input file.")

    client = genai.Client(api_key=api_key)
    results = []
    for index, model in enumerate(MODELS):
        results.append(benchmark_model(client, model, source, QUESTION_COUNT))
        if index < len(MODELS) - 1:
            time.sleep(MODEL_DELAY_SECONDS)
    save_results(results, Path(OUTPUT_FILE), Path(OUTPUT_CSV_FILE))
    print(json.dumps(results, indent=2))
    print(f"JSON results saved to: {OUTPUT_FILE}")
    print(f"CSV results saved to: {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    main()
