"""
tests/conftest.py — shared fixture builders for the decision-engine and
injection-resistance test suites.

These builders construct Stage 1/2/3 objects directly (IngestedClaim,
ClaimExtraction, SubClaim, ImageFinding, UserHistoryRecord) rather than reading
dataset/sample_claims.csv or calling any LLM. Stage 4 (decision_engine.decide)
takes exactly these objects as input and makes no LLM call itself, so it is fully
testable this way — see ARCHITECTURE_AND_STRATEGY.md §9 ("write one pytest case
per labeled row ... given the (mocked) Stage 2/3 outputs").

None of the values below are copied from dataset/sample_claims.csv or
dataset/claims.csv — every scenario is a synthetic case written to exercise one
specific rule in decision_engine.py, consistent with "do not hardcode case IDs,
labels, or answers" in the production code. Naming convention used throughout:
the dataset's row-by-row patterns inspired *which rules* to test, not the literal
fixture content.

Run with: `python -m pytest tests/ -v` from the project root. (Plain `pytest` may
fail to import `src.*` depending on your pytest version's import-mode defaults —
`python -m pytest` guarantees the project root is on sys.path.)
"""

from __future__ import annotations

import pytest

from src.schemas import (
    ClaimExtraction,
    EvidenceRequirement,
    ImageFinding,
    IngestedClaim,
    SubClaim,
    UserHistoryRecord,
)


def make_sub_claim(
    object_part_claimed: str = "rear_bumper",
    issue_family: str = "dent_or_scratch",
    severity_language: str = "small mark",
) -> SubClaim:
    return SubClaim(
        object_part_claimed=object_part_claimed,
        issue_family=issue_family,
        severity_language=severity_language,
    )


def make_claim_extraction(
    primary_claim: SubClaim | None = None,
    secondary_claims: list[SubClaim] | None = None,
    injection_attempts_detected: list[str] | None = None,
    claimed_vehicle_identity_hint: str | None = None,
    extraction_confidence: str = "high",
) -> ClaimExtraction:
    return ClaimExtraction(
        primary_claim=primary_claim or make_sub_claim(),
        secondary_claims=secondary_claims or [],
        injection_attempts_detected=injection_attempts_detected or [],
        claimed_vehicle_identity_hint=claimed_vehicle_identity_hint,
        extraction_confidence=extraction_confidence,
    )


def make_image_finding(
    image_id: str = "img_1",
    observed_object_class: str = "car",
    observed_object_part: str = "rear_bumper",
    observed_issue_type: str = "dent",
    issue_visually_confirmed: bool = True,
    part_visible: bool = True,
    quality_flags: list[str] | None = None,
    authenticity_flags: list[str] | None = None,
    embedded_text_detected: str | None = None,
    observed_vehicle_color: str | None = None,
    severity_estimate: str = "medium",
    rationale: str = "Synthetic test fixture.",
) -> ImageFinding:
    return ImageFinding(
        image_id=image_id,
        observed_object_class=observed_object_class,
        observed_object_part=observed_object_part,
        observed_issue_type=observed_issue_type,
        issue_visually_confirmed=issue_visually_confirmed,
        part_visible=part_visible,
        quality_flags=quality_flags or [],
        authenticity_flags=authenticity_flags or [],
        embedded_text_detected=embedded_text_detected,
        observed_vehicle_color=observed_vehicle_color,
        severity_estimate=severity_estimate,
        rationale=rationale,
    )


def make_user_history(
    user_id: str = "test_user",
    history_flags: list[str] | None = None,
    history_summary: str = "Synthetic test fixture history.",
) -> UserHistoryRecord:
    return UserHistoryRecord(
        user_id=user_id,
        past_claim_count=0,
        accept_claim=0,
        manual_review_claim=0,
        rejected_claim=0,
        last_90_days_claim_count=0,
        history_flags=history_flags or [],
        history_summary=history_summary,
    )


def make_ingested_claim(
    user_id: str = "test_user",
    claim_object: str = "car",
    user_claim_raw: str = "Customer: There is a dent on the rear bumper.",
    image_paths: list[str] | None = None,
    image_ids: list[str] | None = None,
    evidence_requirements: list[EvidenceRequirement] | None = None,
    user_history: UserHistoryRecord | None = None,
) -> IngestedClaim:
    return IngestedClaim(
        user_id=user_id,
        claim_object=claim_object,
        user_claim_raw=user_claim_raw,
        image_paths=image_paths or ["images/test/synthetic/img_1.jpg"],
        image_ids=image_ids or ["img_1"],
        evidence_requirements=evidence_requirements or [],
        user_history=user_history if user_history is not None else make_user_history(user_id=user_id),
    )


# Fixture wrappers, for tests that prefer pytest's dependency-injection style over a
# plain function import.
@pytest.fixture
def sub_claim_factory():
    return make_sub_claim


@pytest.fixture
def claim_extraction_factory():
    return make_claim_extraction


@pytest.fixture
def image_finding_factory():
    return make_image_finding


@pytest.fixture
def user_history_factory():
    return make_user_history


@pytest.fixture
def ingested_claim_factory():
    return make_ingested_claim