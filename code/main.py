"""
main.py — top-level orchestrator for the multi-modal evidence review pipeline.

Runs Stages 1-5 end to end against dataset/claims.csv and writes output.csv:

    Stage 1 (ingestion)         src.utils.build_ingested_claims
    Stage 2 (claim extraction)  src.claim_extraction.extract_claim   — one call per claim row
    Stage 3 (visual assessment) src.visual_assessment.assess_image   — one call per image
    Stage 4 (decision engine)   src.decision_engine.decide           — pure code, no LLM call
    Stage 5 (output assembly)   assemble_final_row (below)           — maps onto the 14-column schema

Stages 2 and 3 are each run as one flat, bounded-concurrency batch (all claim-extraction
calls together, then all visual-assessment calls together) rather than interleaved
per-claim. This is a deliberate simplification over "fan out per claim, fan out images
within each claim": running the two LLM-call types in two separate phases keeps the
total number of concurrent in-flight requests easy to reason about for TPM/RPM budgeting
(ARCHITECTURE_AND_STRATEGY.md §10), at the cost of slightly higher latency than maximal
interleaving would give. Stage 4 has no LLM call, so it does not affect concurrency.

`run_pipeline()` is the single function reused by both this script and
evaluation/run_eval.py, so the two never run subtly different pipelines.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from src.claim_extraction import extract_claim
from src.decision_engine import decide
from src.history import UserHistoryIndex
from src.schemas import (
    ClaimExtraction,
    DecisionResult,
    FinalOutputRow,
    ImageFinding,
    IngestedClaim,
    PipelineTrace,
    SubClaim,
)
from src.utils import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    JSONCache,
    PROJECT_ROOT,
    build_ingested_claims,
    get_gemini_client,
    index_evidence_requirements,
    load_evidence_requirements,
    run_with_concurrency,
    write_output_csv,
)
from src.visual_assessment import assess_image

logger = logging.getLogger("evidence_review")

DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_CLAIMS_CSV = DEFAULT_DATASET_DIR / "claims.csv"
DEFAULT_EVIDENCE_CSV = DEFAULT_DATASET_DIR / "evidence_requirements.csv"
DEFAULT_HISTORY_CSV = DEFAULT_DATASET_DIR / "user_history.csv"
# image_paths in the CSVs already include the "images/..." prefix, so the root they
# resolve against is the dataset directory itself, not dataset/images.
DEFAULT_IMAGES_ROOT = DEFAULT_DATASET_DIR
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "output.csv"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Stage 5 — output assembly
# ---------------------------------------------------------------------------


def assemble_final_row(ingested: IngestedClaim, decision: DecisionResult) -> FinalOutputRow:
    """Maps Stage 4's DecisionResult onto the exact 14-column output schema. Every
    field here is a direct pass-through from either IngestedClaim (the original row's
    own input fields, echoed back unchanged) or DecisionResult (everything the
    decision engine computed) — no new judgment happens at this stage."""
    return FinalOutputRow(
        user_id=ingested.user_id,
        image_paths=";".join(ingested.image_paths),
        user_claim=ingested.user_claim_raw,
        claim_object=ingested.claim_object,
        evidence_standard_met=decision.evidence_standard_met,
        evidence_standard_met_reason=decision.evidence_standard_met_reason,
        risk_flags=decision.risk_flags,
        issue_type=decision.issue_type,
        object_part=decision.object_part,
        claim_status=decision.claim_status,
        claim_status_justification=decision.claim_status_justification,
        supporting_image_ids=decision.supporting_image_ids,
        valid_image=decision.valid_image,
        severity=decision.severity,
    )


def _emergency_claim_extraction_fallback() -> ClaimExtraction:
    """Used only if a Stage 2 task raises *past* claim_extraction.py's own internal
    fallback (e.g. an exception in run_with_concurrency's gather itself, not in the
    Gemini call). Mirrors the same "degrade towards not_enough_information" policy as
    claim_extraction._conservative_fallback, duplicated narrowly here rather than
    imported, since that helper is private to its module by design."""
    return ClaimExtraction(
        primary_claim=SubClaim(
            object_part_claimed="unknown",
            issue_family="unclear",
            severity_language="unknown",
        ),
        extraction_confidence="low",
    )


# ---------------------------------------------------------------------------
# Usage / operational-analysis tracking (ARCHITECTURE_AND_STRATEGY.md §10)
# ---------------------------------------------------------------------------


class UsageTracker:
    """Accumulates per-stage call/token/cache counters for the operational analysis
    report. Updated only from the main coroutine after each gather() completes —
    never mutated concurrently from inside a worker task."""

    def __init__(self) -> None:
        self.total_claims = 0
        self.claim_extraction_calls = 0
        self.claim_extraction_cache_hits = 0
        self.visual_assessment_calls = 0
        self.visual_assessment_cache_hits = 0
        self.images_processed = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.claim_extraction_seconds = 0.0
        self.visual_assessment_seconds = 0.0
        self.total_seconds = 0.0

    def add_claim_usage(self, usage: dict) -> None:
        self.claim_extraction_calls += 1
        if usage.get("cache_hit"):
            self.claim_extraction_cache_hits += 1
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)

    def add_image_usage(self, usage: dict) -> None:
        self.visual_assessment_calls += 1
        self.images_processed += 1
        if usage.get("cache_hit"):
            self.visual_assessment_cache_hits += 1
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)

    def as_dict(self) -> dict:
        return {
            "total_claims": self.total_claims,
            "claim_extraction_calls": self.claim_extraction_calls,
            "claim_extraction_cache_hits": self.claim_extraction_cache_hits,
            "visual_assessment_calls": self.visual_assessment_calls,
            "visual_assessment_cache_hits": self.visual_assessment_cache_hits,
            "images_processed": self.images_processed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "claim_extraction_seconds": round(self.claim_extraction_seconds, 2),
            "visual_assessment_seconds": round(self.visual_assessment_seconds, 2),
            "total_seconds": round(self.total_seconds, 2),
        }


def print_usage_summary(usage: UsageTracker) -> None:
    d = usage.as_dict()
    logger.info(
        "Run summary: %d claims | claim-extraction calls=%d (cache hits=%d, %.1fs) | "
        "visual-assessment calls=%d on %d images (cache hits=%d, %.1fs) | "
        "tokens in=%d out=%d total=%d | wall clock=%.1fs",
        d["total_claims"],
        d["claim_extraction_calls"],
        d["claim_extraction_cache_hits"],
        d["claim_extraction_seconds"],
        d["visual_assessment_calls"],
        d["images_processed"],
        d["visual_assessment_cache_hits"],
        d["visual_assessment_seconds"],
        d["input_tokens"],
        d["output_tokens"],
        d["total_tokens"],
        d["total_seconds"],
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration (Stages 1-5)
# ---------------------------------------------------------------------------


async def run_pipeline(
    claims_csv: str | Path,
    evidence_csv: str | Path,
    history_csv: str | Path,
    images_root: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    use_cache: bool = True,
) -> tuple[list[PipelineTrace], UsageTracker]:
    """Runs Stages 1-5 over every row of `claims_csv`, in CSV row order. Works
    identically for dataset/claims.csv (no label columns) and dataset/sample_claims.csv
    (label columns present but unused here — see src/utils.build_ingested_claims),
    which is what lets evaluation/run_eval.py reuse this function unmodified.

    Returns (traces, usage) where `traces` is one PipelineTrace per input row, in the
    same order as the input CSV, and `usage` holds aggregate call/token/timing counters
    for the operational analysis report.
    """
    usage = UsageTracker()

    # --- Stage 1: ingestion -------------------------------------------------
    requirements = load_evidence_requirements(evidence_csv)
    evidence_index = index_evidence_requirements(requirements)
    history_index = UserHistoryIndex.load(history_csv)

    # UserHistoryIndex implements .get(user_id) with a conservative no-history
    # default for unknown users (see src/history.py), so it can be passed directly
    # wherever build_ingested_claims expects a `.get`-compatible history lookup.
    ingested_claims = build_ingested_claims(claims_csv, evidence_index, history_index)
    usage.total_claims = len(ingested_claims)
    logger.info("Stage 1: ingested %d claim rows from %s", len(ingested_claims), claims_csv)

    client = get_gemini_client()
    claim_cache = JSONCache(namespace="claim_extraction") if use_cache else None
    image_cache = JSONCache(namespace="visual_assessment") if use_cache else None

    # --- Stage 2: structured claim extraction (one call per claim) ---------
    stage2_start = time.perf_counter()
    logger.info("Stage 2: extracting structured claims for %d rows (concurrency=%d)", len(ingested_claims), concurrency)
    claim_tasks = [
        extract_claim(client, c.claim_object, c.user_claim_raw, cache=claim_cache, model=model)
        for c in ingested_claims
    ]
    claim_results = await run_with_concurrency(claim_tasks, limit=concurrency)

    extractions: list[ClaimExtraction] = []
    for ingested, result in zip(ingested_claims, claim_results):
        if isinstance(result, BaseException):
            logger.error("Claim extraction task raised for user_id=%s: %s", ingested.user_id, result)
            extractions.append(_emergency_claim_extraction_fallback())
            continue
        extraction, claim_usage = result
        usage.add_claim_usage(claim_usage)
        extractions.append(extraction)
    usage.claim_extraction_seconds = time.perf_counter() - stage2_start

    # --- Stage 3: per-image visual assessment (one call per image) ---------
    stage3_start = time.perf_counter()
    image_tasks = []
    image_task_owner: list[int] = []  # index into ingested_claims, one per task
    for idx, c in enumerate(ingested_claims):
        for path, image_id in zip(c.image_paths, c.image_ids):
            image_tasks.append(
                assess_image(
                    client,
                    c.claim_object,
                    image_id,
                    path,
                    images_root,
                    c.evidence_requirements,
                    cache=image_cache,
                    model=model,
                )
            )
            image_task_owner.append(idx)

    logger.info("Stage 3: assessing %d images (concurrency=%d)", len(image_tasks), concurrency)
    image_results = await run_with_concurrency(image_tasks, limit=concurrency)

    findings_by_claim: list[list[ImageFinding]] = [[] for _ in ingested_claims]
    for owner_idx, image_id, result in zip(
        image_task_owner,
        (iid for c in ingested_claims for iid in c.image_ids),
        image_results,
    ):
        if isinstance(result, BaseException):
            # assess_image() already catches and falls back internally for ordinary
            # failures (missing file, API error, schema failure); reaching this branch
            # means the *task itself* raised outside that handling (e.g. cancellation).
            logger.error("Visual assessment task raised for image_id=%s: %s", image_id, result)
            continue
        finding, image_usage = result
        usage.add_image_usage(image_usage)
        findings_by_claim[owner_idx].append(finding)
    usage.visual_assessment_seconds = time.perf_counter() - stage3_start

    # --- Stage 4 (decision engine, pure code) + Stage 5 (output assembly) --
    logger.info("Stage 4-5: running decision engine and assembling output rows")
    traces: list[PipelineTrace] = []
    for ingested, extraction, findings in zip(ingested_claims, extractions, findings_by_claim):
        decision = decide(ingested, extraction, findings)
        output_row = assemble_final_row(ingested, decision)
        traces.append(
            PipelineTrace(
                ingested=ingested,
                claim_extraction=extraction,
                image_findings=findings,
                decision=decision,
                output_row=output_row,
            )
        )

    return traces, usage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Stage 1-5 multi-modal evidence review pipeline over claims.csv."
    )
    parser.add_argument("--claims-csv", default=str(DEFAULT_CLAIMS_CSV), help="Input claims CSV.")
    parser.add_argument("--evidence-csv", default=str(DEFAULT_EVIDENCE_CSV), help="evidence_requirements.csv path.")
    parser.add_argument("--history-csv", default=str(DEFAULT_HISTORY_CSV), help="user_history.csv path.")
    parser.add_argument(
        "--images-root",
        default=str(DEFAULT_IMAGES_ROOT),
        help="Directory that image_paths in the CSV are relative to (default: dataset/).",
    )
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Where to write predictions.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name (default: gemini-2.5-flash).")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent LLM calls.")
    parser.add_argument("--no-cache", action="store_true", help="Disable the disk-backed JSON cache.")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, or ERROR.")
    return parser


async def _main_async(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    started = time.perf_counter()

    traces, usage = await run_pipeline(
        claims_csv=args.claims_csv,
        evidence_csv=args.evidence_csv,
        history_csv=args.history_csv,
        images_root=args.images_root,
        model=args.model,
        concurrency=args.concurrency,
        use_cache=not args.no_cache,
    )
    usage.total_seconds = time.perf_counter() - started

    output_rows = [t.output_row for t in traces]
    write_output_csv(output_rows, args.output_csv)
    logger.info("Wrote %d rows to %s", len(output_rows), args.output_csv)

    print_usage_summary(usage)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()