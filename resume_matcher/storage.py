"""Simple JSON-file persistence for candidate records and the active job description.

Good enough for a single-recruiter local session; swap for a real database
if this ever needs concurrent multi-user access.
"""
import json
import logging
import os
import threading

import config

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CANDIDATES_FILE = os.path.join(config.DATA_DIR, "candidates.json")
_JD_FILE = os.path.join(config.DATA_DIR, "job_description.json")


def _read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return default


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_candidate(candidate):
    """Insert or replace a candidate record, keyed by candidate['id']."""
    with _LOCK:
        candidates = _read_json(_CANDIDATES_FILE, {})
        candidates[candidate["id"]] = candidate
        _write_json(_CANDIDATES_FILE, candidates)


def list_candidates():
    return list(_read_json(_CANDIDATES_FILE, {}).values())


def clear_candidates():
    """Remove every stored candidate record."""
    with _LOCK:
        _write_json(_CANDIDATES_FILE, {})


def save_job_description(jd_record):
    with _LOCK:
        _write_json(_JD_FILE, jd_record)


def get_job_description():
    return _read_json(_JD_FILE, None)
