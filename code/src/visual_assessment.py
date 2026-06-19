"""
Stage 3 — Per-image visual assessment (VLM call, run once per image, parallelizable).

For each image independently: object/part identification, issue/condition observed,
image quality flags, authenticity signals, and verbatim transcription of any in-image
text. This call is NEVER shown the claim text — see ARCHITECTURE_AND_STRATEGY.md §4
and §5: it only receives `claim_object` (to know which part vocabulary applies) and
the relevant evidence_requirements.csv rows (to know what a thorough inspection should
consider), never "the user says the hood is scratched." Running this per-image rather
than batching all of a claim's images into one call is deliberate (§4): it is what
makes per-image `supporting_image_ids` selection (pick the clear image, not the blurry
one) defensible and auditable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from src.schemas import EvidenceRequirement, ImageFinding
from src.utils import (
    DEFAULT_MODEL,
    JSONCache,
    SchemaValidationFailed,
    call_gemini_structured,
    compute_bytes_hash,
    get_allowed_issue_types,
    get_allowed_object_parts,
    guess_mime_type,
    read_image_bytes,
)

logger = logging.getLogger("evidence_review")

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "visual_assessment.txt"
_PROMPT_TEMPLATE = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _format_evidence_lines(evidence_requirements: list[EvidenceRequirement]) -> str:
    if not evidence_requirements:
        return "- (no specific evidence requirement rows matched this object category)"
    return "\n".join(
        f"- {req.requirement_id} ({req.applies_to}): {req.minimum_image_evidence}"
        for req in evidence_requirements
    )


def build_visual_assessment_prompt(
    claim_object: str,
    image_id: str,
    evidence_requirements: list[EvidenceRequirement],
) -> str:
    return _PROMPT_TEMPLATE.format(
        claim_object=claim_object,
        image_id=image_id,
        allowed_object_parts=", ".join(get_allowed_object_parts(claim_object)),
        allowed_issue_types=", ".join(get_allowed_issue_types()),
        evidence_requirement_lines=_format_evidence_lines(evidence_requirements),
    )


def _conservative_fallback(image_id: str, reason: str) -> ImageFinding:
    """Used when the image can't be read at all, or the LLM call never produces
    schema-valid JSON. Mirrors §10's fallback philosophy: report the part as not
    visible/inspectable rather than guessing — this naturally pushes the decision
    engine's sufficiency gate towards `not_enough_information` instead of fabricating
    a verdict from no evidence."""
    logger.error("Visual assessment fallback for image_id=%s: %s", image_id, reason)
    return ImageFinding(
        image_id=image_id,
        observed_object_class="unclear",
        observed_object_part="unknown",
        observed_issue_type="unknown",
        issue_visually_confirmed=False,
        part_visible=False,
        quality_flags=[],
        authenticity_flags=[],
        embedded_text_detected=None,
        observed_vehicle_color=None,
        observed_vehicle_orientation=None,
        severity_estimate="unknown",
        rationale=f"Image could not be assessed ({reason}); treated as not inspectable.",
    )


async def assess_image(
    client: genai.Client,
    claim_object: str,
    image_id: str,
    image_path: str,
    images_root: str | Path,
    evidence_requirements: list[EvidenceRequirement],
    cache: JSONCache | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[ImageFinding, dict]:
    """Runs Stage 3 for one image. Returns (ImageFinding, usage_info)."""
    image_bytes = read_image_bytes(image_path, images_root)
    if image_bytes is None:
        return _conservative_fallback(image_id, "image file not found on disk"), {
            "input_tokens": 0,
            "output_tokens": 0,
            "attempts": 0,
            "cache_hit": False,
        }

    # Content-hash keyed cache (§10), with claim_object folded in: the same bytes
    # should not be reused across two different claim_object contexts, since the
    # prompt's vocabulary hints (and therefore a model's best answer) differ by
    # claim_object even if the pixels happen to collide.
    cache_key = f"{compute_bytes_hash(image_bytes)}_{claim_object}"

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            try:
                finding = ImageFinding.model_validate_json(cached)
                finding.image_id = image_id  # cache key is content-based, not id-based
                return finding, {"input_tokens": 0, "output_tokens": 0, "attempts": 0, "cache_hit": True}
            except Exception as e:
                logger.warning("Ignoring corrupt visual-assessment cache entry %s: %s", cache_key, e)

    prompt = build_visual_assessment_prompt(
        claim_object=claim_object, image_id=image_id, evidence_requirements=evidence_requirements
    )
    image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=guess_mime_type(image_path))

    try:
        result, usage_info = await call_gemini_structured(
            client=client,
            model=model,
            contents=[prompt, image_part],
            response_schema=ImageFinding,
        )
    except SchemaValidationFailed as e:
        result = _conservative_fallback(image_id, str(e))
        usage_info = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    except Exception as e:
        result = _conservative_fallback(image_id, f"Unexpected error calling Gemini: {e}")
        usage_info = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}

    result.image_id = image_id  # guard against the model echoing a different id
    usage_info["cache_hit"] = False

    if cache is not None:
        cache.set(cache_key, result.model_dump_json())

    return result, usage_info