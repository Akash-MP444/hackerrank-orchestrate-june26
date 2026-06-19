"""
Stage 2 — Structured claim extraction (text-only LLM call).

Turns the raw, possibly multilingual, possibly distractor-laden, possibly adversarial
conversation transcript into a `ClaimExtraction` object. This call is NEVER shown any
image — see ARCHITECTURE_AND_STRATEGY.md §4 and §5: the claim-extraction model must
have no way to let visual content leak into its description of what was *said*, and
the decision engine (Stage 4) is what compares the two independently-produced
descriptions.

This module's only job is to produce a faithful `ClaimExtraction` (the model's
self-report, including its own self-report of injection attempts). It does NOT run
the independent heuristic injection scanner — that happens in decision_engine.py,
which has access to both this output and the raw transcript, and is the single place
where `risk_flags` is decided. Keeping the heuristic check there (not here) keeps
"what the model said" and "what the system decided" cleanly separated.
"""

from __future__ import annotations

import logging
from pathlib import Path

from google import genai

from src.schemas import ClaimExtraction, SubClaim
from src.utils import (
    DEFAULT_MODEL,
    JSONCache,
    SchemaValidationFailed,
    call_gemini_structured,
    compute_text_hash,
)

logger = logging.getLogger("evidence_review")

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "claim_extraction.txt"
_PROMPT_TEMPLATE = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def build_claim_extraction_prompt(claim_object: str, conversation: str) -> str:
    return _PROMPT_TEMPLATE.format(claim_object=claim_object, conversation=conversation)


def _conservative_fallback(reason: str) -> ClaimExtraction:
    """Used only when the LLM call fails outright (transient-error budget exhausted)
    or never produces schema-valid JSON even after the validation-retry. Mirrors
    ARCHITECTURE_AND_STRATEGY.md §10's fallback philosophy: degrade towards
    "not enough information" rather than crash the row or silently invent a verdict.
    Note this does NOT suppress injection detection — decision_engine.py's independent
    heuristic scan over the raw transcript still runs regardless of whether this
    fallback was used.
    """
    logger.error("Claim extraction fallback triggered: %s", reason)
    return ClaimExtraction(
        primary_claim=SubClaim(
            object_part_claimed="unknown",
            issue_family="unclear",
            severity_language="unknown",
        ),
        secondary_claims=[],
        ruled_out_topics=[],
        detected_languages=["unknown"],
        user_scoping_instructions=[],
        injection_attempts_detected=[],
        claimed_vehicle_identity_hint=None,
        extraction_confidence="low",
    )


async def extract_claim(
    client: genai.Client,
    claim_object: str,
    user_claim_raw: str,
    cache: JSONCache | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[ClaimExtraction, dict]:
    """Runs Stage 2 for one claim. Returns (ClaimExtraction, usage_info).

    `usage_info` is `{"input_tokens": int, "output_tokens": int, "attempts": int,
    "cache_hit": bool}` for cost/latency reporting in evaluation/run_eval.py.
    """
    cache_key = compute_text_hash(f"{claim_object}_{user_claim_raw}")

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            try:
                return ClaimExtraction.model_validate_json(cached), {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "attempts": 0,
                    "cache_hit": True,
                }
            except Exception as e:  # cache corruption shouldn't break the pipeline
                logger.warning("Ignoring corrupt claim-extraction cache entry %s: %s", cache_key, e)

    prompt = build_claim_extraction_prompt(claim_object=claim_object, conversation=user_claim_raw)

    try:
        result, usage_info = await call_gemini_structured(
            client=client,
            model=model,
            contents=[prompt],
            response_schema=ClaimExtraction,
        )
    except SchemaValidationFailed as e:
        result = _conservative_fallback(str(e))
        usage_info = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    except Exception as e:  # any other unexpected API failure: don't crash the row
        result = _conservative_fallback(f"Unexpected error calling Gemini: {e}")
        usage_info = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}

    usage_info["cache_hit"] = False

    if cache is not None:
        cache.set(cache_key, result.model_dump_json())

    return result, usage_info