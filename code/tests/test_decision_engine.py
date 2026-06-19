"""
tests/test_decision_engine.py — unit tests for src/decision_engine.py, the one
module in the pipeline with zero LLM calls and therefore the one place a test
suite can assert exact, deterministic behavior.

Each test below targets one named rule from ARCHITECTURE_AND_STRATEGY.md (cited
in the docstring) using a synthetic fixture, NOT a row copied from
dataset/sample_claims.csv. The goal is to lock in the *behavioral pattern* the
labelers used, not to memorize any specific dataset answer.

NOTE on get_issue_family(): several tests below rely on
config/allowed_values.yaml's issue_family_map mapping issue_type strings like
"dent"/"scratch" -> "dent_or_scratch" and "crack" -> "crack_or_glass" (the natural
reading of the IssueFamily literal names in src/schemas.py). If your actual
config file maps these differently, the issue_family-mismatch-related tests
(test_supported_when_part_and_issue_match, test_contradicted_severity_mismatch)
may need their `issue_family=` fixture values adjusted to match.
"""

from __future__ import annotations

from conftest import (
    make_claim_extraction,
    make_image_finding,
    make_ingested_claim,
    make_sub_claim,
    make_user_history,
)

from src.decision_engine import decide


def test_supported_when_part_and_issue_match():
    """§1.1/§1.3 baseline: claimed part visible, observed issue matches the claimed
    issue family, no high-severity language mismatch -> supported, with the visible
    image cited as the sole supporting image and an empty risk_flags set."""
    ingested = make_ingested_claim(claim_object="car", image_ids=["img_1"])
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(
            object_part_claimed="door", issue_family="dent_or_scratch", severity_language="small dent"
        )
    )
    findings = [
        make_image_finding(
            image_id="img_1",
            observed_object_class="car",
            observed_object_part="door",
            observed_issue_type="dent",
            severity_estimate="medium",
        )
    ]

    result = decide(ingested, claim, findings)

    assert result.claim_status == "supported"
    assert result.evidence_standard_met is True
    assert result.valid_image is True
    assert result.issue_type == "dent"
    assert result.object_part == "door"
    assert result.severity == "medium"
    assert result.supporting_image_ids == ["img_1"]
    assert result.risk_flags == []


def test_not_enough_information_when_claimed_part_not_visible():
    """§1.1/§1.4: the claimed part is simply absent from frame (part_visible=False).
    This must be not_enough_information, not contradicted — and per
    decision_engine.best_guess_part, object_part falls back to the claimed text
    itself ('headlight'), not a bare 'unknown', when no finding even attempted to
    identify that part."""
    ingested = make_ingested_claim(claim_object="car")
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(
            object_part_claimed="headlight", issue_family="crack_or_glass", severity_language="small crack"
        )
    )
    findings = [
        make_image_finding(
            image_id="img_1",
            observed_object_class="car",
            observed_object_part="rear_bumper",
            observed_issue_type="unknown",
            part_visible=False,
            severity_estimate="unknown",
        )
    ]

    result = decide(ingested, claim, findings)

    assert result.claim_status == "not_enough_information"
    assert result.evidence_standard_met is False
    assert result.object_part == "headlight"  # falls back to claimed text, not "unknown"
    assert result.issue_type == "unknown"
    assert result.severity == "unknown"
    assert result.supporting_image_ids == []
    assert "damage_not_visible" in result.risk_flags


def test_contradicted_severity_mismatch():
    """§1.3 sub-pattern 'severity mismatch': the claim's own wording implies serious
    damage, but the visible severity is light. object_part and issue_family both
    match the claim -- only the *severity* claim is contradicted."""
    ingested = make_ingested_claim(claim_object="car")
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(
            object_part_claimed="rear_bumper",
            issue_family="dent_or_scratch",
            severity_language="badly damaged",
        )
    )
    findings = [
        make_image_finding(
            image_id="img_1",
            observed_object_class="car",
            observed_object_part="rear_bumper",
            observed_issue_type="scratch",
            severity_estimate="low",
        )
    ]

    result = decide(ingested, claim, findings)

    assert result.claim_status == "contradicted"
    assert "claim_mismatch" in result.risk_flags
    assert "manual_review_required" in result.risk_flags
    assert result.issue_type == "scratch"
    assert result.severity == "low"


def test_contradicted_wrong_object_class():
    """§1.3 sub-pattern 'wrong object entirely': the photographed object doesn't even
    match the claimed object category. object_part/issue_type collapse to
    'unknown' and both wrong_object and claim_mismatch are raised."""
    ingested = make_ingested_claim(claim_object="package")
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(
            object_part_claimed="box", issue_family="torn_or_crushed_packaging", severity_language="crushed"
        )
    )
    findings = [
        make_image_finding(
            image_id="img_1",
            observed_object_class="car",
            observed_object_part="front_bumper",
            observed_issue_type="dent",
            severity_estimate="medium",
        )
    ]

    result = decide(ingested, claim, findings)

    assert result.claim_status == "contradicted"
    assert result.object_part == "unknown"
    assert result.issue_type == "unknown"
    assert {"wrong_object", "claim_mismatch", "manual_review_required"} <= set(result.risk_flags)


def test_contradicted_no_issue_found():
    """§1.3 sub-pattern 'damage claimed, none visible': the claimed part IS visible
    and identified, but shows no issue at all (observed_issue_type='none'). This is
    a contradiction, not not_enough_information, per §1.4's discriminator."""
    ingested = make_ingested_claim(claim_object="laptop")
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(
            object_part_claimed="trackpad",
            issue_family="mechanical_malfunction",
            severity_language="stopped working",
        )
    )
    findings = [
        make_image_finding(
            image_id="img_1",
            observed_object_class="laptop",
            observed_object_part="trackpad",
            observed_issue_type="none",
            severity_estimate="none",
        )
    ]

    result = decide(ingested, claim, findings)

    assert result.claim_status == "contradicted"
    assert result.issue_type == "none"
    assert "damage_not_visible" in result.risk_flags
    assert "manual_review_required" in result.risk_flags


def test_pick_clearest_image_among_multiple():
    """§4/§9: when one submitted image is unusable (blurry) but another is clean,
    only the clean image should be cited in supporting_image_ids — but the quality
    flag from the unusable image should still surface in risk_flags."""
    ingested = make_ingested_claim(claim_object="car", image_ids=["img_1", "img_2"])
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(object_part_claimed="rear_bumper", issue_family="dent_or_scratch")
    )
    blurry = make_image_finding(
        image_id="img_1",
        observed_object_part="rear_bumper",
        observed_issue_type="dent",
        issue_visually_confirmed=False,
        quality_flags=["blurry_image"],
        severity_estimate="low",
    )
    clean = make_image_finding(
        image_id="img_2",
        observed_object_part="rear_bumper",
        observed_issue_type="dent",
        issue_visually_confirmed=True,
        quality_flags=[],
        severity_estimate="medium",
    )

    result = decide(ingested, claim, [blurry, clean])

    assert result.claim_status == "supported"
    assert result.supporting_image_ids == ["img_2"]
    assert "blurry_image" in result.risk_flags
    assert "manual_review_required" not in result.risk_flags


def test_valid_image_independent_of_evidence_standard_met():
    """§1.2: valid_image (authenticity/usability) and evidence_standard_met (content
    sufficiency) are independent gates. An authenticity flag should drag valid_image
    to false and force manual_review_required, without affecting evidence_standard_met
    or claim_status when the visible content otherwise supports the claim."""
    ingested = make_ingested_claim(claim_object="car")
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(object_part_claimed="door", issue_family="dent_or_scratch")
    )
    findings = [
        make_image_finding(
            image_id="img_1",
            observed_object_part="door",
            observed_issue_type="dent",
            authenticity_flags=["non_original_image"],
            severity_estimate="medium",
        )
    ]

    result = decide(ingested, claim, findings)

    assert result.evidence_standard_met is True
    assert result.claim_status == "supported"
    assert result.valid_image is False
    assert "non_original_image" in result.risk_flags
    assert "manual_review_required" in result.risk_flags


def test_user_history_risk_does_not_override_clear_visual_evidence():
    """§5.3, the single most important guarantee in the system: user history can add
    a risk flag and a justification sentence, but it must never flip claim_status by
    itself. Here the visual evidence is unambiguously supporting, and the user has a
    risky history -- the verdict must still be 'supported'."""
    risky_history = make_user_history(
        history_flags=["user_history_risk"],
        history_summary="History note marker for this test case.",
    )
    ingested = make_ingested_claim(claim_object="car", user_history=risky_history)
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(object_part_claimed="door", issue_family="dent_or_scratch")
    )
    findings = [
        make_image_finding(image_id="img_1", observed_object_part="door", observed_issue_type="dent")
    ]

    result = decide(ingested, claim, findings)

    assert result.claim_status == "supported"
    assert "user_history_risk" in result.risk_flags
    assert "manual_review_required" in result.risk_flags
    assert "History note marker for this test case." in result.claim_status_justification


def test_secondary_claim_image_does_not_pollute_primary_evaluation():
    """§1.7/§6.4 multi-issue handling: an image that clearly depicts a named
    SECONDARY sub-claim's part must be excluded from evaluating the PRIMARY claim
    (matches_domain), so a multi-part claim's secondary photo can't accidentally
    become the 'best' evidence for the primary issue. The secondary issue should
    still be named in the justification text."""
    ingested = make_ingested_claim(claim_object="car", image_ids=["img_1", "img_2"])
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(object_part_claimed="rear_bumper", issue_family="dent_or_scratch"),
        secondary_claims=[
            make_sub_claim(
                object_part_claimed="headlight",
                issue_family="broken_or_missing_part",
                severity_language="broken",
            )
        ],
    )
    headlight_finding = make_image_finding(
        image_id="img_1", observed_object_part="headlight", observed_issue_type="broken_part"
    )
    bumper_finding = make_image_finding(
        image_id="img_2", observed_object_part="rear_bumper", observed_issue_type="dent", severity_estimate="medium"
    )

    result = decide(ingested, claim, [headlight_finding, bumper_finding])

    assert result.claim_status == "supported"
    assert result.object_part == "rear_bumper"
    assert result.supporting_image_ids == ["img_2"]
    assert "headlight" in result.claim_status_justification


def test_vehicle_identity_mismatch_is_flagged_but_currently_does_not_alone_change_the_verdict():
    """§1.7, REQ_CAR_IDENTITY_OR_SIDE -- documents CURRENT (as-shipped) behavior, and
    flags a real gap found while writing this test.

    _vehicle_identity_mismatch() correctly detects a claimed-vs-observed color
    mismatch and adds 'vehicle_identity_mismatch' + 'claim_mismatch' to risk_flags.
    However, that detection happens independently of the `is_mismatch` boolean that
    actually sets claim_status, and 'claim_mismatch'/'vehicle_identity_mismatch'
    being present in risk_flags is NOT one of the manual_review_required trigger
    conditions on its own (only claim_status == 'contradicted' is). So today, a
    vehicle-identity mismatch is recorded but does not by itself flip claim_status
    away from 'supported' or force manual_review_required.

    This test locks in that current behavior so it doesn't silently change, and
    exists to flag it as a follow-up: consider folding vehicle_identity_mismatch
    into the `is_mismatch` expression (or adding it to the manual_review_required
    trigger set) in decision_engine.decide() if a color/orientation mismatch should
    be treated as seriously as the other claim_mismatch sub-patterns.
    """
    ingested = make_ingested_claim(claim_object="car")
    claim = make_claim_extraction(
        primary_claim=make_sub_claim(object_part_claimed="front_bumper", issue_family="dent_or_scratch"),
        claimed_vehicle_identity_hint="my blue car",
    )
    findings = [
        make_image_finding(
            image_id="img_1",
            observed_object_part="front_bumper",
            observed_issue_type="dent",
            observed_vehicle_color="black",
            severity_estimate="medium",
        )
    ]

    result = decide(ingested, claim, findings)

    assert "vehicle_identity_mismatch" in result.risk_flags
    assert "claim_mismatch" in result.risk_flags
    # Documents the gap: as currently implemented, neither of the above flips the verdict.
    assert result.claim_status == "supported"
    assert "manual_review_required" not in result.risk_flags