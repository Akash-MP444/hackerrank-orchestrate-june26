"""
Shared infrastructure for the evidence-review pipeline:

- Paths / env / config loading (config/allowed_values.yaml, config/pricing.yaml)
- CSV reading/writing for claims.csv / sample_claims.csv / output.csv
- Stage 1 ingestion helpers (joining a raw row against evidence_requirements.csv and
  user_history.csv into an IngestedClaim)
- A small disk-backed JSON cache, keyed by content hash, used by both Stage 2 and
  Stage 3 to avoid re-paying for LLM calls on unchanged input (see
  ARCHITECTURE_AND_STRATEGY.md §10, "Caching")
- A deterministic, non-LLM prompt-injection heuristic scanner — defense-in-depth, so
  injection resistance does not rely solely on the model choosing to self-report
- The async google-genai call wrapper: tenacity-backed retry for transient errors,
  plus the "retry-with-validation-error-appended, then fall back conservatively"
  strategy described in §10 ("Retry/validation strategy")

No case IDs, user IDs, or specific labels are referenced anywhere in this module.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Optional, Type, TypeVar

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.schemas import EvidenceRequirement, IngestedClaim, UserHistoryRecord

logger = logging.getLogger("evidence_review")

load_dotenv()

ModelT = TypeVar("ModelT", bound=BaseModel)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "10"))

CSV_OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


# ---------------------------------------------------------------------------
# Config / vocabulary loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_ALLOWED_VALUES_CACHE: Optional[dict] = None


def _allowed_values() -> dict:
    global _ALLOWED_VALUES_CACHE
    if _ALLOWED_VALUES_CACHE is None:
        _ALLOWED_VALUES_CACHE = _load_yaml(CONFIG_DIR / "allowed_values.yaml")
    return _ALLOWED_VALUES_CACHE


def get_allowed_object_parts(claim_object: str) -> list[str]:
    return list(_allowed_values().get("object_parts", {}).get(claim_object, ["unknown"]))


def get_allowed_issue_types() -> list[str]:
    return list(_allowed_values().get("issue_types", ["none", "unknown"]))


def get_issue_family(issue_type: str) -> str:
    """Maps a normalized issue_type string to schemas.IssueFamily. Defaults to
    'unclear' for anything not in the config rather than raising, since the held-out
    test set may use vocabulary the 20-row sample didn't exercise."""
    mapping = _allowed_values().get("issue_family_map", {})
    return mapping.get(normalize_token(issue_type), "unclear")


def load_pricing() -> dict:
    return _load_yaml(CONFIG_DIR / "pricing.yaml")


# ---------------------------------------------------------------------------
# Generic text normalization
# ---------------------------------------------------------------------------


def normalize_token(value: Optional[str]) -> str:
    """Lowercase, strip, collapse internal whitespace/punctuation to single
    underscores. Used to compare claimed vs. observed object_part / issue_type
    without being defeated by 'Rear Bumper' vs 'rear_bumper' vs 'rear-bumper'."""
    if not value:
        return ""
    v = value.strip().lower()
    v = re.sub(r"[\s\-/]+", "_", v)
    v = re.sub(r"[^a-z0-9_]", "", v)
    v = re.sub(r"_+", "_", v).strip("_")
    return v


# ---------------------------------------------------------------------------
# CSV reading / writing
# ---------------------------------------------------------------------------


def read_csv_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def derive_image_id(image_path: str) -> str:
    """'images/test/case_001/img_2.jpg' -> 'img_2'."""
    stem = Path(image_path).stem
    return stem


def load_evidence_requirements(path: str | Path) -> list[EvidenceRequirement]:
    return [EvidenceRequirement(**row) for row in read_csv_rows(path)]


def index_evidence_requirements(
    requirements: list[EvidenceRequirement],
) -> dict[str, list[EvidenceRequirement]]:
    """claim_object -> the requirements that apply to it, including every row whose
    claim_object is 'all'."""
    index: dict[str, list[EvidenceRequirement]] = {}
    all_rows = [r for r in requirements if r.claim_object == "all"]
    for obj in {r.claim_object for r in requirements if r.claim_object != "all"}:
        index[obj] = [r for r in requirements if r.claim_object == obj] + all_rows
    return index


def load_user_history(path: str | Path) -> dict[str, UserHistoryRecord]:
    records = [UserHistoryRecord(**row) for row in read_csv_rows(path)]
    return {r.user_id: r for r in records}


def build_ingested_claims(
    csv_path: str | Path,
    evidence_index: dict[str, list[EvidenceRequirement]],
    history_index: dict[str, UserHistoryRecord],
) -> list[IngestedClaim]:
    """Stage 1: load a claims CSV (claims.csv or sample_claims.csv — both share the
    first 4 columns) and join each row against reference data. Label columns present
    in sample_claims.csv (evidence_standard_met, claim_status, ...) are ignored here;
    evaluation/run_eval.py reads them separately for scoring."""
    claims: list[IngestedClaim] = []
    for row in read_csv_rows(csv_path):
        image_paths = [p.strip() for p in row["image_paths"].split(";") if p.strip()]
        claims.append(
            IngestedClaim(
                user_id=row["user_id"],
                claim_object=row["claim_object"],
                user_claim_raw=row["user_claim"],
                image_paths=image_paths,
                image_ids=[derive_image_id(p) for p in image_paths],
                evidence_requirements=evidence_index.get(row["claim_object"], []),
                user_history=history_index.get(row["user_id"]),
            )
        )
    return claims


def _bool_to_csv(value: bool) -> str:
    return "true" if value else "false"


def _list_to_csv(values: list[str]) -> str:
    return ";".join(values) if values else "none"


def output_row_to_csv_dict(row) -> dict:
    """Converts a schemas.FinalOutputRow into the exact string-typed dict shape used
    by sample_claims.csv / output.csv (semicolon-joined lists, lowercase booleans)."""
    return {
        "user_id": row.user_id,
        "image_paths": row.image_paths,
        "user_claim": row.user_claim,
        "claim_object": row.claim_object,
        "evidence_standard_met": _bool_to_csv(row.evidence_standard_met),
        "evidence_standard_met_reason": row.evidence_standard_met_reason,
        "risk_flags": _list_to_csv(row.risk_flags),
        "issue_type": row.issue_type,
        "object_part": row.object_part,
        "claim_status": row.claim_status,
        "claim_status_justification": row.claim_status_justification,
        "supporting_image_ids": _list_to_csv(row.supporting_image_ids),
        "valid_image": _bool_to_csv(row.valid_image),
        "severity": row.severity,
    }


def write_output_csv(rows: list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(output_row_to_csv_dict(row))


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def read_image_bytes(image_path: str, images_root: str | Path) -> Optional[bytes]:
    """Resolves image_path (as written in the CSV, e.g. 'images/test/case_001/img_1.jpg')
    relative to images_root and reads it. Returns None — never raises — when the file
    is missing, so the caller can fall back to a conservative ImageFinding instead of
    crashing the whole row (see ARCHITECTURE_AND_STRATEGY.md §10)."""
    full_path = Path(images_root) / image_path
    if not full_path.is_file():
        logger.warning("Image file not found, skipping vision call: %s", full_path)
        return None
    return full_path.read_bytes()


def guess_mime_type(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    return mime or "image/jpeg"


# ---------------------------------------------------------------------------
# Disk-backed JSON cache, keyed by content hash (image bytes or normalized text)
# ---------------------------------------------------------------------------


def compute_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class JSONCache:
    """A flat-file cache: one JSON file per key under `directory/{namespace}/{key}.json`.

    Keying by content hash (not by row index, user_id, or path) means re-running the
    pipeline during development, or hitting the same image/transcript twice, never
    re-pays for an LLM call — see ARCHITECTURE_AND_STRATEGY.md §10.
    """

    def __init__(self, directory: str | Path = DEFAULT_CACHE_DIR, namespace: str = "default"):
        self.dir = Path(directory) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        p = self._path(key)
        if not p.is_file():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None

    def set(self, key: str, value: str) -> None:
        try:
            self._path(key).write_text(value, encoding="utf-8")
        except OSError as e:
            logger.warning("Cache write failed for key %s: %s", key, e)


# ---------------------------------------------------------------------------
# Deterministic prompt-injection heuristic scanner (defense in depth)
# ---------------------------------------------------------------------------

# Each pattern targets the *structure* of an override attempt (directive verbs aimed
# at an automated system: ignore/disregard/override prior instructions, command the
# system to mark/approve/treat the claim a certain way, claims of elevated authority,
# or instructions to reveal/change system behavior) rather than any one case's exact
# wording, so this generalizes beyond the single planted test-set example.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|the|previous|prior|above)[\w\s]{0,30}instructions?", re.I),
    re.compile(r"disregard (all|any|the|previous|prior|above)[\w\s]{0,30}(instructions?|rules?)", re.I),
    re.compile(r"(mark|set|treat|flag|label)\s+(this|it|the claim|the row)\s+as\s+\w+", re.I),
    re.compile(r"\boverride\b.{0,30}(instructions?|system|decision|verdict|output)", re.I),
    re.compile(r"you are (now|actually)\s+\w+", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"\bact as\b.{0,30}(admin|developer|system|reviewer)", re.I),
    re.compile(r"\bdo not (flag|review|evaluate|check)\b", re.I),
    re.compile(r"\bskip\b.{0,20}(review|evaluation|check|verification)", re.I),
    re.compile(r"\bapprove\b.{0,20}(this|the)\s+(claim|case|row)", re.I),
    re.compile(r"\bbypass\b.{0,20}(review|check|verification|evidence)", re.I),
]


def detect_injection_heuristics(text: str) -> list[str]:
    """Deterministic, non-LLM scan for prompt-injection-shaped phrases in raw text.

    This exists so injection resistance does not rely solely on the claim-extraction
    model choosing to self-report a directive in `injection_attempts_detected` — a
    model that is itself fooled would simply omit it. The decision engine unions this
    function's output with the model's self-report (see decision_engine.py)."""
    if not text:
        return []
    hits: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        for match in pattern.finditer(text):
            snippet = match.group(0).strip()
            if snippet and snippet not in hits:
                hits.append(snippet)
    return hits


# ---------------------------------------------------------------------------
# Gemini client + structured-output call wrapper
# ---------------------------------------------------------------------------


class SchemaValidationFailed(Exception):
    """Raised when a structured LLM response failed Pydantic validation even after
    the one allowed reprompt-with-error-message retry. Callers should catch this and
    fall back to a conservative default rather than crash the row."""


_CLIENT: Optional[genai.Client] = None


def get_gemini_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or a .env file."
            )
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _generate_content_with_backoff(
    client: genai.Client,
    model: str,
    contents: list,
    config: genai_types.GenerateContentConfig,
) -> genai_types.GenerateContentResponse:
    """Tenacity-wrapped call for *transient* failures (429 rate limits, 5xx server
    errors, timeouts). Schema-validation failures are handled separately by
    call_gemini_structured, since those need the validation error appended to the
    prompt before retrying, not a blind retry of the same input."""
    return await client.aio.models.generate_content(model=model, contents=contents, config=config)


async def call_gemini_structured(
    client: genai.Client,
    model: str,
    contents: list,
    response_schema: Type[ModelT],
    *,
    temperature: float = 0.0,
    max_validation_retries: int = 1,
) -> tuple[ModelT, dict]:
    """Calls Gemini with a Pydantic response_schema and returns a validated instance.

    Retry strategy (ARCHITECTURE_AND_STRATEGY.md §10):
      1. Transient errors (429/5xx/timeout) are retried with exponential backoff by
         `_generate_content_with_backoff` (tenacity), invisibly to this function.
      2. If the response doesn't validate against `response_schema`, we append the
         validation error text to the prompt and ask once more (`max_validation_retries`).
      3. If still invalid, raise SchemaValidationFailed — the caller is responsible for
         falling back to a conservative default rather than crashing the row.

    Returns (validated_instance, usage_info) where usage_info has token counts for
    cost/latency reporting in evaluation/run_eval.py.
    """
    current_contents = list(contents)
    last_error: Optional[Exception] = None
    usage_info: dict = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}

    for attempt in range(max_validation_retries + 1):
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
        )
        try:
            response = await _generate_content_with_backoff(client, model, current_contents, config)
        except RetryError as e:
            raise SchemaValidationFailed(f"Transient API failure after retries: {e}") from e

        usage_info["attempts"] += 1
        if response.usage_metadata is not None:
            usage_info["input_tokens"] += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            usage_info["output_tokens"] += (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )

        # Prefer the SDK's own parsed instance; fall back to manual validation of the
        # raw text in case the SDK couldn't fully populate `.parsed` (e.g. an unrelated
        # extra field, or a future SDK version that changes this behavior).
        if isinstance(response.parsed, response_schema):
            return response.parsed, usage_info

        raw_text = response.text or ""
        try:
            validated = response_schema.model_validate_json(raw_text)
            return validated, usage_info
        except ValidationError as e:
            last_error = e
            logger.warning(
                "Structured output failed validation (attempt %d/%d): %s",
                attempt + 1,
                max_validation_retries + 1,
                e,
            )
            current_contents = [
                *contents,
                (
                    f"\n\nYour previous output did not match the required schema. "
                    f"Validation error: {e}\nReturn corrected JSON only, no prose."
                ),
            ]

    raise SchemaValidationFailed(
        f"Schema validation failed after {max_validation_retries + 1} attempt(s): {last_error}"
    )


# ---------------------------------------------------------------------------
# Concurrency helper
# ---------------------------------------------------------------------------


async def run_with_concurrency(coroutines: list, limit: int = DEFAULT_CONCURRENCY) -> list:
    """Runs all coroutines with a bounded concurrency, preserving input order in the
    output list. Exceptions are returned in-place (not raised) so one failed call
    doesn't abort the whole batch — callers should check `isinstance(r, Exception)`."""
    semaphore = asyncio.Semaphore(limit)

    async def _bounded(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_bounded(c) for c in coroutines), return_exceptions=True)