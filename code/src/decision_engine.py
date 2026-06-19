"""
Stage 4 — Deterministic decision engine. NO LLM call happens in this module.

Takes the Stage 2 output (`ClaimExtraction`), the list of Stage 3 outputs
(`ImageFinding`, one per image), and the Stage 1 reference joins already attached to
`IngestedClaim` (evidence_requirements, user_history), and applies an explicit rule
table to compute every output column. The LLM never directly emits `claim_status`,
`evidence_standard_met`, or `manual_review_required` — it emits observations; this
module emits decisions. Every output field below is traceable to either a named rule
here or a specific Stage 2/3 field — see the inline comments citing
ARCHITECTURE_AND_STRATEGY.md section numbers.

Two places where this implementation makes an explicit, documented choice beyond the
architecture doc's §6.4 pseudocode sketch (which the doc itself only presents as a
"pseudocode" skeleton, and which has at least one internal inconsistency — see
schemas.py's RiskFlag docstring):

1. `text_instruction_present` vs `injection_attempt_detected` — kept as two distinct
   flags fed by two distinct sources (image-embedded text vs. conversational
   injection), per §1.5/§1.6's prose rather than §6.4's pseudocode line.
2. Severity-language mismatch ("claims it's badly damaged, image shows a light
   scratch") is not addressed by any formula in §6.4, but is explicitly named as a
   real sub-pattern in §1.3. Since `severity_language` is free text, detecting this
   purely deterministically requires a keyword heuristic (`_claims_high_severity`)
   — an acknowledged approximation, consistent with the architecture doc's own
   instruction (§11 talking point #8) to name heuristic limitations rather than
   hide them.
"""

from __future__ import annotations

from src.schemas import ClaimExtraction, DecisionResult, ImageFinding, IngestedClaim, SubClaim
from src.utils import detect_injection_heuristics, get_issue_family, normalize_token

# Keyword approximation for "the user's own words imply serious damage" — used only to
# detect the severity-mismatch sub-pattern from §1.3 ("claims 'damaged', image shows a
# light scratch"). Deliberately broad/lowercase-substring rather than tied to any
# specific case's exact wording.
_HIGH_SEVERITY_KEYWORDS = [
    "bad",
    "badly",
    "severe",
    "severely",
    "completely",
    "totally",
    "destroyed",
    "shattered",
    "smashed",
    "wrecked",
    "extensive",
    "major",
    "significant",
    "significantly",
    "heavily",
    "seriously",
    "awful",
    "terrible",
    "ruined",
    "huge",
]

_COLOR_WORDS = [
    "red", "blue", "black", "white", "silver", "grey", "gray",
    "green", "yellow", "brown", "orange", "maroon",
]


# ---------------------------------------------------------------------------
# Small pure helpers (each independently unit-testable — see tests/test_decision_engine.py)
# ---------------------------------------------------------------------------


def matches_domain(finding: ImageFinding, claim: ClaimExtraction) -> bool:
    """A finding is 'in-domain' for evaluating the PRIMARY claim unless it clearly,
    specifically depicts a different, named SECONDARY sub-claim's part instead. This
    routes a multi-part claim's images to the right sub-claim without excluding a
    wrong-object/wrong-part image — which must stay in-domain so it can surface as a
    contradiction (§1.3 'wrong object entirely') rather than be silently dropped into
    'not enough information'."""
    if not claim.secondary_claims:
        return True
    primary_norm = normalize_token(claim.primary_claim.object_part_claimed)
    finding_norm = normalize_token(finding.observed_object_part)
    if not finding_norm:
        return True
    for secondary in claim.secondary_claims:
        secondary_norm = normalize_token(secondary.object_part_claimed)
        if secondary_norm and finding_norm == secondary_norm and finding_norm != primary_norm:
            return False
    return True


def pick_clearest(visible: list[ImageFinding]) -> ImageFinding:
    """Prefers, in order: no quality flags, a visually-confirmed issue, a non-
    'unknown' observed_issue_type. Ties broken by image_id for determinism (same
    Stage 2/3 inputs must always produce the same Stage 4 output — see
    ARCHITECTURE_AND_STRATEGY.md §4, 'Determinism under re-run')."""

    def score(f: ImageFinding) -> tuple:
        return (
            0 if not f.quality_flags else 1,
            0 if f.issue_visually_confirmed else 1,
            0 if normalize_token(f.observed_issue_type) not in ("", "unknown") else 1,
        )

    return sorted(visible, key=lambda f: (score(f), f.image_id))[0]


def best_guess_part(findings: list[ImageFinding], primary: SubClaim) -> str | None:
    """Used only on the not_enough_information path. Prefers a finding that, even
    though not confidently `part_visible`, still named the same part as the claim
    (the model attempted an identification but couldn't confirm it). Falls back to
    the claimed part text itself — "what we were trying to evaluate" — rather than a
    bare 'unknown', since that is strictly more useful to a human reviewer."""
    claimed_norm = normalize_token(primary.object_part_claimed)
    if claimed_norm and claimed_norm != "unknown":
        for f in findings:
            if normalize_token(f.observed_object_part) == claimed_norm:
                return f.observed_object_part
        return primary.object_part_claimed
    return None


def _claims_high_severity(severity_language: str) -> bool:
    text = (severity_language or "").lower()
    return any(kw in text for kw in _HIGH_SEVERITY_KEYWORDS)


def _vehicle_identity_mismatch(
    claim: ClaimExtraction, visible: list[ImageFinding], claim_object: str
) -> bool:
    """REQ_CAR_IDENTITY_OR_SIDE (§1.7): only fires when the user's own wording named a
    color, AND at least one visible image actually reports an observed color to
    compare against, AND none of those observed colors match. Silent (no mismatch)
    whenever there's nothing concrete to compare — this check should never manufacture
    a finding from absence of information."""
    if claim_object != "car" or not claim.claimed_vehicle_identity_hint:
        return False
    hint = claim.claimed_vehicle_identity_hint.lower()
    claimed_colors = [c for c in _COLOR_WORDS if c in hint]
    if not claimed_colors:
        return False
    observed_colors = {
        normalize_token(f.observed_vehicle_color) for f in visible if f.observed_vehicle_color
    }
    if not observed_colors:
        return False
    return not any(c in observed_colors for c in claimed_colors)


# ---------------------------------------------------------------------------
# Justification text builders — grounded in actual field values, never templated
# boilerplate detached from the row's own evidence (§2.4: "grounded justifications").
# ---------------------------------------------------------------------------


def _evidence_reason(visible: list[ImageFinding], primary: SubClaim) -> str:
    if visible:
        ids = ", ".join(f.image_id for f in visible)
        return (
            f"The claimed {primary.object_part_claimed} is visible and inspectable in "
            f"image(s) {ids}, so the claim can be evaluated."
        )
    return (
        f"The claimed {primary.object_part_claimed} is not visible or inspectable in "
        f"any submitted image, so the claim cannot be evaluated."
    )


def _claim_status_justification(
    claim_status: str,
    primary: SubClaim,
    best: ImageFinding | None,
    visible: list[ImageFinding],
    mismatch_reasons: list[str],
    secondary_claims: list[SubClaim],
    history_note: str | None,
    injection_note: str | None,
) -> str:
    if claim_status == "not_enough_information":
        sentence = (
            f"None of the submitted images show the claimed {primary.object_part_claimed} "
            f"clearly enough to confirm or deny the claim."
        )
    elif best is None:  # defensive; should not happen when claim_status != not_enough_information
        sentence = "No visual finding was available to ground a verdict."
    elif claim_status == "supported":
        sentence = (
            f"Image {best.image_id} shows {best.observed_object_part} with "
            f"{best.observed_issue_type} consistent with the claimed "
            f"{primary.object_part_claimed} damage."
        )
    else:  # contradicted
        reason_text = "; ".join(mismatch_reasons) if mismatch_reasons else "the claim does not match what is visible"
        sentence = (
            f"Image(s) {', '.join(f.image_id for f in visible)} show "
            f"{best.observed_object_part} ({best.observed_issue_type}), which does not "
            f"support the claimed {primary.object_part_claimed} report: {reason_text}."
        )

    if secondary_claims:
        names = ", ".join(s.object_part_claimed for s in secondary_claims)
        sentence += f" The claim also separately mentions {names}, which is not the primary issue evaluated here."
    if history_note:
        sentence += f" {history_note}"
    if injection_note:
        sentence += f" {injection_note}"
    return sentence


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def decide(
    ingested: IngestedClaim,
    claim: ClaimExtraction,
    findings: list[ImageFinding],
) -> DecisionResult:
    """The single function that produces every decision-layer output column."""
    primary = claim.primary_claim
    claim_object = ingested.claim_object

    relevant = [f for f in findings if matches_domain(f, claim)]
    visible = [f for f in relevant if f.part_visible]

    # --- Authenticity / usability gate: independent of content sufficiency (§1.2) ---
    valid_image = not any(f.authenticity_flags for f in findings)

    # --- Risk flags from the vision pass (quality + authenticity), unioned over ALL
    #     submitted images, not just the ones used to ground the verdict (§1.5a) ---
    risk_flags: set[str] = set()
    for f in findings:
        risk_flags.update(f.quality_flags)
        risk_flags.update(f.authenticity_flags)

    # --- text_instruction_present: image-embedded text only (§1.5) ---
    if any((f.embedded_text_detected or "").strip() for f in findings):
        risk_flags.add("text_instruction_present")

    # --- injection_attempt_detected: conversational injection, detected via the
    #     UNION of Stage 2's self-report AND an independent deterministic heuristic
    #     scan over the raw transcript (§1.6) — never trusting the LLM's self-report
    #     alone, since a model that was successfully fooled would simply omit it. ---
    heuristic_hits = detect_injection_heuristics(ingested.user_claim_raw)
    injection_detected = bool(claim.injection_attempts_detected) or bool(heuristic_hits)
    if injection_detected:
        risk_flags.add("injection_attempt_detected")

    injection_note = None
    if injection_detected:
        evidence = list(dict.fromkeys([*claim.injection_attempts_detected, *heuristic_hits]))
        injection_note = (
            f"A prompt-injection-style instruction was detected in the submitted text "
            f"({'; '.join(evidence[:2])}) and was disregarded; it did not influence "
            f"this verdict."
        )

    # --- vehicle identity / orientation consistency (§1.7, REQ_CAR_IDENTITY_OR_SIDE) ---
    if _vehicle_identity_mismatch(claim, visible, claim_object):
        risk_flags.add("vehicle_identity_mismatch")
        risk_flags.add("claim_mismatch")

    mismatch_reasons: list[str] = []

    if not visible:
        # --- Sufficiency gate: nothing usable to evaluate against (§1.1, §1.4) ---
        evidence_standard_met = False
        claim_status = "not_enough_information"
        object_part = best_guess_part(findings, primary) or "unknown"
        issue_type = "unknown"
        severity = "unknown"
        supporting_ids: list[str] = []
        best: ImageFinding | None = None
        risk_flags.add("damage_not_visible")
    else:
        evidence_standard_met = True
        best = pick_clearest(visible)

        class_mismatch = normalize_token(best.observed_object_class) != normalize_token(claim_object)
        object_part_mismatch = normalize_token(best.observed_object_part) != normalize_token(
            primary.object_part_claimed
        )
        no_issue_found = (
            not class_mismatch
            and normalize_token(best.observed_issue_type) == "none"
            and primary.issue_family != "unclear"
        )
        observed_family = get_issue_family(best.observed_issue_type)
        issue_family_mismatch = (
            not class_mismatch
            and not no_issue_found
            and primary.issue_family != "unclear"
            and observed_family != primary.issue_family
        )
        severity_mismatch = (
            not class_mismatch
            and not no_issue_found
            and _claims_high_severity(primary.severity_language)
            and best.severity_estimate in ("none", "low")
        )

        is_mismatch = class_mismatch or object_part_mismatch or no_issue_found or issue_family_mismatch or severity_mismatch

        if class_mismatch:
            risk_flags.update({"wrong_object", "claim_mismatch"})
            object_part, issue_type = "unknown", "unknown"
            mismatch_reasons.append(
                f"the submitted image shows a {best.observed_object_class}, not the claimed {claim_object}"
            )
        else:
            object_part, issue_type = best.observed_object_part, best.observed_issue_type
            if object_part_mismatch:
                risk_flags.add("claim_mismatch")
                mismatch_reasons.append(
                    f"the visible part ({best.observed_object_part}) differs from the claimed part "
                    f"({primary.object_part_claimed})"
                )
            if issue_family_mismatch:
                risk_flags.add("claim_mismatch")
                mismatch_reasons.append(
                    f"the visible issue ({best.observed_issue_type}) differs from the claimed issue family"
                )
            if severity_mismatch:
                risk_flags.add("claim_mismatch")
                mismatch_reasons.append(
                    "the claim's wording implies serious damage but the visible severity is "
                    f"{best.severity_estimate}"
                )
            if no_issue_found:
                risk_flags.add("damage_not_visible")
                mismatch_reasons.append(
                    f"the claimed area is visible but shows no {primary.issue_family.replace('_', ' ')}"
                )

        claim_status = "contradicted" if is_mismatch else "supported"
        severity = best.severity_estimate
        supporting_ids = [best.image_id] if claim_status == "supported" else [f.image_id for f in visible]

    # --- User history: contributes a risk flag + justification sentence ONLY. It is
    #     structurally excluded from every condition above that sets `claim_status` —
    #     this is the literal code-level enforcement of §5.3 ("history cannot override
    #     clear visual evidence by itself"). ---
    history_note = None
    if ingested.user_history is not None and ingested.user_history.has_risk_signal:
        risk_flags.add("user_history_risk")
        history_note = f"User history note: {ingested.user_history.history_summary}."

    # --- manual_review_required: DERIVED, never primary (§1.5b). Also escalated for
    #     a detected injection attempt — a deliberate, documented extension beyond
    #     §6.4's trigger list, since adversarial input warrants human eyes regardless
    #     of how the verdict itself came out. ---
    if (
        claim_status == "contradicted"
        or "user_history_risk" in risk_flags
        or bool(risk_flags & {"possible_manipulation", "non_original_image"})
        or not evidence_standard_met
        or "injection_attempt_detected" in risk_flags
    ):
        risk_flags.add("manual_review_required")

    evidence_reason = _evidence_reason(visible, primary)
    justification = _claim_status_justification(
        claim_status=claim_status,
        primary=primary,
        best=best,
        visible=visible,
        mismatch_reasons=mismatch_reasons,
        secondary_claims=claim.secondary_claims,
        history_note=history_note,
        injection_note=injection_note,
    )

    return DecisionResult(
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason=evidence_reason,
        risk_flags=sorted(risk_flags),
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=supporting_ids,
        valid_image=valid_image,
        severity=severity,
    )