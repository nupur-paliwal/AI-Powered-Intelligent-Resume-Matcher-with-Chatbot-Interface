"""Flask application: resume upload/processing, JD intake, ranking, and chat."""
import logging
import os
import uuid

from flask import Flask, jsonify, render_template, request

import config
from resume_matcher import chatbot, jd_parser, resume_parser, scoring, storage, vector_store
from resume_matcher.extraction import ExtractionError, extract_text
from resume_matcher.llm_provider import llm_backend_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = config.FLASK_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH_MB * 1024 * 1024


def _allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in config.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template(
        "index.html",
        llm_backend=llm_backend_name(),
        min_resumes=config.MIN_RESUMES_RECOMMENDED,
    )


@app.route("/api/status")
def status():
    return jsonify(
        {
            "llm_backend": llm_backend_name(),
            "candidate_count": len(storage.list_candidates()),
            "job_description_set": storage.get_job_description() is not None,
        }
    )


@app.route("/api/upload_resumes", methods=["POST"])
def upload_resumes():
    files = request.files.getlist("resumes")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    results = [_process_resume(file) for file in files]

    jd_record = storage.get_job_description()
    if jd_record:
        _rescore_all(jd_record)

    return jsonify({"results": results, "candidates": _ranked_candidates()})


def _process_resume(file):
    filename = file.filename or "unknown.pdf"
    if not _allowed_file(filename):
        return {
            "filename": filename,
            "status": "error",
            "message": "Only PDF files are supported.",
        }

    candidate_id = uuid.uuid4().hex
    file_path = os.path.join(config.UPLOAD_DIR, f"{candidate_id}_{filename}")
    file.save(file_path)

    try:
        extraction = extract_text(file_path)
    except ExtractionError as exc:
        return {"filename": filename, "status": "error", "message": str(exc)}

    if not extraction["text"].strip():
        return {
            "filename": filename,
            "status": "error",
            "message": "No text could be extracted, even with OCR.",
            "warnings": extraction["warnings"],
        }

    fields = resume_parser.parse_resume(extraction["text"])
    vector_store.index_resume(candidate_id, filename, extraction["text"])

    candidate = {
        "id": candidate_id,
        "filename": filename,
        "text": extraction["text"],
        "extraction_method": extraction["method"],
        "warnings": extraction["warnings"],
        "fields": fields,
        "score": None,
    }
    storage.save_candidate(candidate)

    return {
        "filename": filename,
        "status": "success",
        "extraction_method": extraction["method"],
        "warnings": extraction["warnings"],
        "fields": fields,
    }


@app.route("/api/job_description", methods=["POST"])
def submit_job_description():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Job description text is required."}), 400

    fields = jd_parser.parse_job_description(text)
    jd_record = {"text": text, "fields": fields}
    storage.save_job_description(jd_record)
    _rescore_all(jd_record)

    return jsonify({"fields": fields, "candidates": _ranked_candidates()})


def _rescore_all(jd_record):
    for candidate in storage.list_candidates():
        candidate["score"] = scoring.compute_match_score(
            candidate["text"], candidate["fields"], jd_record["text"], jd_record["fields"]
        )
        storage.save_candidate(candidate)


def _ranked_candidates():
    candidates = sorted(
        storage.list_candidates(),
        key=lambda c: (c["score"] or {}).get("overall", -1),
        reverse=True,
    )
    return [
        {
            "id": c["id"],
            "filename": c["filename"],
            "extraction_method": c["extraction_method"],
            "fields": c["fields"],
            "score": c["score"],
        }
        for c in candidates
    ]


@app.route("/api/candidates")
def get_candidates():
    return jsonify({"candidates": _ranked_candidates()})


@app.route("/api/candidates", methods=["DELETE"])
def clear_candidates():
    for candidate in storage.list_candidates():
        file_path = os.path.join(config.UPLOAD_DIR, f"{candidate['id']}_{candidate['filename']}")
        if os.path.exists(file_path):
            os.remove(file_path)

    vector_store.clear_all()
    storage.clear_candidates()

    return jsonify({"candidates": []})


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query is required."}), 400
    return jsonify({"answer": chatbot.answer_query(query)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
