"""
evaluation/run_eval.py — runs the full Stage 1-5 pipeline against
dataset/sample_claims.csv and scores the predictions against the ground-truth columns
already present in that file.

This module never hardcodes a case ID, a user ID, or any specific expected value —
every label it scores against is read from sample_claims.csv at runtime, and the one
adversarial-input regression check (injection_regression_check) is driven by the same
deterministic heuristic scanner the production pipeline itself uses
(src.utils.detect_injection_heuristics), not by knowing which row the injection lives in.

Reuses main.run_pipeline() rather than re-implementing Stage 1-5, so the sample-set
evaluation and the real claims.csv run are guaranteed to be the same pipeline.

Outputs (written under evaluation/ by default):
  - sample_predictions.csv   the pipeline's own output.csv-shaped predictions for
                              every row in sample_claims.csv
  - error_analysis.csv       one row per (claim, mismatched field), expected vs.
                              predicted, with enough Stage 2/3 context to debug it
  - evaluation_report.md     metrics + the operational analysis section required by
                              problem_statement.md ("Evaluation requirement")
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (  # noqa: E402  (path must be set up before this import)
    DEFAULT_EVIDENCE_CSV,
    DEFAULT_HISTORY_CSV,
    DEFAULT_IMAGES_ROOT,
    UsageTracker,
    configure_logging,
    run_pipeline,
)
from src.schemas import PipelineTrace  # noqa: E402
from src.utils import (  # noqa: E402
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    PROJECT_ROOT as SRC_PROJECT_ROOT,
    detect_injection_heuristics,
    load_pricing,
    read_csv_rows,
    write_output_csv,
)

logger = logging.getLogger("evidence_review.eval")

DEFAULT_SAMPLE_CSV = SRC_PROJECT_ROOT / "dataset" / "sample_claims.csv"
DEFAULT_EVAL_OUTPUT_DIR = SRC_PROJECT_ROOT / "evaluation"

# Fields scored by exact string/bool match. claim_status is reported separately too
# (confusion matrix) since its error *direction* matters more than its raw rate for a
# system like this one — see ARCHITECTURE_AND_STRATEGY.md §9.
EXACT_MATCH_FIELDS = [
    "claim_status",
    "evidence_standard_met",
    "valid_image",
    "issue_type",
    "object_part",
    "severity",
]

# Fields scored as sets: order doesn't matter, and a system that finds the same
# *content* in a different sequence should not be penalized as if it were wrong.
SET_FIELDS = ["risk_flags", "supporting_image_ids"]


# ---------------------------------------------------------------------------
# Label loading (read from sample_claims.csv at runtime — never hardcoded)
# ---------------------------------------------------------------------------


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _parse_semicolon_set(value: str) -> set[str]:
    v = (value or "").strip()
    if not v or v.lower() == "none":
        return set()
    return {part.strip() for part in v.split(";") if part.strip()}


def load_labels(sample_csv: str | Path) -> list[dict]:
    """Returns one label dict per row of sample_claims.csv, in file order. Row order
    is the join key back to the pipeline's traces — both iterate the same CSV in the
    same order via src.utils.read_csv_rows / build_ingested_claims."""
    labels = []
    for row in read_csv_rows(sample_csv):
        labels.append(
            {
                "evidence_standard_met": _parse_bool(row["evidence_standard_met"]),
                "valid_image": _parse_bool(row["valid_image"]),
                "claim_status": row["claim_status"].strip(),
                "issue_type": row["issue_type"].strip(),
                "object_part": row["object_part"].strip(),
                "severity": row["severity"].strip(),
                "risk_flags": _parse_semicolon_set(row["risk_flags"]),
                "supporting_image_ids": _parse_semicolon_set(row["supporting_image_ids"]),
            }
        )
    return labels


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score(traces: list[PipelineTrace], labels: list[dict]) -> dict:
    """Per-field accuracy, set-overlap metrics, a row-level exact-match rate, and a
    claim_status confusion matrix. Returns a single plain-dict report consumed by both
    write_error_analysis_csv and write_evaluation_report_md."""
    n = min(len(traces), len(labels))
    if len(traces) != len(labels):
        logger.warning(
            "Pipeline produced %d rows but sample_claims.csv has %d label rows; "
            "scoring only the first %d.",
            len(traces),
            len(labels),
            n,
        )

    field_correct: Counter = Counter()
    field_total: Counter = Counter()
    set_jaccard_sum: Counter = Counter()
    confusion: Counter = Counter()
    row_exact_count = 0
    per_row_results: list[dict] = []

    for i in range(n):
        trace, label = traces[i], labels[i]
        row = trace.output_row
        mismatches: dict[str, dict] = {}

        for field in EXACT_MATCH_FIELDS:
            field_total[field] += 1
            predicted = getattr(row, field)
            expected = label[field]
            is_match = (
                predicted == expected
                if isinstance(expected, bool)
                else str(predicted).strip().lower() == str(expected).strip().lower()
            )
            if is_match:
                field_correct[field] += 1
            else:
                mismatches[field] = {"expected": expected, "predicted": predicted}

        confusion[(label["claim_status"], row.claim_status)] += 1

        for field in SET_FIELDS:
            field_total[field] += 1
            predicted_set = set(getattr(row, field))
            expected_set = label[field]
            overlap = _jaccard(predicted_set, expected_set)
            set_jaccard_sum[field] += overlap
            if overlap >= 1.0:
                field_correct[field] += 1
            else:
                mismatches[field] = {
                    "expected": sorted(expected_set),
                    "predicted": sorted(predicted_set),
                    "jaccard": round(overlap, 2),
                }

        row_exact = not mismatches
        if row_exact:
            row_exact_count += 1

        per_row_results.append(
            {
                "row_index": i,
                "user_id": trace.ingested.user_id,
                "claim_object": trace.ingested.claim_object,
                "row_exact_match": row_exact,
                "mismatches": mismatches,
                "claim_extraction_confidence": trace.claim_extraction.extraction_confidence,
                "image_ids": trace.ingested.image_ids,
            }
        )

    return {
        "n_rows": n,
        "row_exact_match_rate": (row_exact_count / n) if n else 0.0,
        "field_accuracy": {f: (field_correct[f] / field_total[f]) for f in EXACT_MATCH_FIELDS},
        "field_set_exact_match_rate": {f: (field_correct[f] / field_total[f]) for f in SET_FIELDS},
        "field_set_mean_jaccard": {f: (set_jaccard_sum[f] / field_total[f]) for f in SET_FIELDS},
        "claim_status_confusion": {f"{expected}->{predicted}": c for (expected, predicted), c in confusion.items()},
        "per_row_results": per_row_results,
    }


def injection_regression_check(traces: list[PipelineTrace]) -> dict:
    """A structural pass/fail check, independent of sample_claims.csv's labels: for
    every row whose RAW transcript trips the deterministic injection heuristic
    (the same src.utils.detect_injection_heuristics the production decision engine
    itself calls), assert the pipeline actually flagged 'injection_attempt_detected'
    in its own output. This stays meaningful on the held-out claims.csv too, since it
    never references a case ID, a user_id, or a hand-labeled answer — it only checks
    that the pipeline's behavior is consistent with its own detector."""
    flagged_user_ids: list[str] = []
    failures: list[dict] = []

    for trace in traces:
        hits = detect_injection_heuristics(trace.ingested.user_claim_raw)
        if not hits:
            continue
        flagged_user_ids.append(trace.ingested.user_id)
        if "injection_attempt_detected" not in trace.output_row.risk_flags:
            failures.append(
                {
                    "user_id": trace.ingested.user_id,
                    "detected_phrases": hits,
                    "output_risk_flags": trace.output_row.risk_flags,
                    "output_claim_status": trace.output_row.claim_status,
                }
            )

    return {
        "rows_with_injection_language": len(flagged_user_ids),
        "user_ids_with_injection_language": flagged_user_ids,
        "rows_where_pipeline_missed_it": failures,
        "passed": len(failures) == 0,
    }


# ---------------------------------------------------------------------------
# Operational analysis / cost estimate (ARCHITECTURE_AND_STRATEGY.md §10)
# ---------------------------------------------------------------------------


def estimate_cost(usage: UsageTracker, pricing: dict, model: str) -> dict:
    """Looks up $/1M-token rates for `model` from config/pricing.yaml. Tolerant of an
    unknown or differently-shaped pricing file: falls back to a clearly-labeled zero
    estimate rather than crashing the eval run, since this is a reporting nice-to-have,
    not something that should block producing output.csv.

    Expected (but not required) pricing.yaml shape:
        models:
          gemini-2.5-flash:
            input_per_million_usd: <float>
            output_per_million_usd: <float>
    """
    model_pricing = (pricing.get("models") or {}).get(model) or {}
    input_rate = model_pricing.get("input_per_million_usd")
    output_rate = model_pricing.get("output_per_million_usd")

    if input_rate is None or output_rate is None:
        return {
            "model": model,
            "pricing_found": False,
            "note": (
                f"No pricing entry for model '{model}' found in config/pricing.yaml "
                "(expected models.<model>.input_per_million_usd / output_per_million_usd). "
                "Reporting token counts only; update pricing.yaml for a dollar estimate."
            ),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "estimated_cost_usd": None,
        }

    input_cost = (usage.input_tokens / 1_000_000) * input_rate
    output_cost = (usage.output_tokens / 1_000_000) * output_rate
    return {
        "model": model,
        "pricing_found": True,
        "input_per_million_usd": input_rate,
        "output_per_million_usd": output_rate,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "estimated_cost_usd": round(input_cost + output_cost, 4),
    }


def project_full_test_cost(sample_usage: UsageTracker, sample_n: int, test_n: int, pricing: dict, model: str) -> dict:
    """Linearly scales the sample-run's measured per-claim and per-image token usage
    up to the full test-set size. A linear projection (rather than re-running against
    claims.csv just to estimate cost) is the right level of precision here — per
    problem_statement.md, this is meant to be an "approximate cost ... with pricing
    assumptions", not a guarantee."""
    if sample_n == 0:
        return {"note": "No sample rows processed; cannot project.", "estimated_cost_usd": None}

    avg_input_per_claim_row = sample_usage.input_tokens / sample_n
    avg_output_per_claim_row = sample_usage.output_tokens / sample_n
    projected_input = avg_input_per_claim_row * test_n
    projected_output = avg_output_per_claim_row * test_n

    projected_usage = UsageTracker()
    projected_usage.input_tokens = int(projected_input)
    projected_usage.output_tokens = int(projected_output)
    cost = estimate_cost(projected_usage, pricing, model)
    cost["projected_from_sample_rows"] = sample_n
    cost["projected_to_test_rows"] = test_n
    cost["assumption"] = (
        "Average tokens-per-claim-row observed on the sample set is assumed to hold for "
        "the full test set. Actual cost will vary with conversation length, image count "
        "per claim, and image resolution."
    )
    return cost


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_sample_predictions_csv(traces: list[PipelineTrace], path: str | Path) -> None:
    write_output_csv([t.output_row for t in traces], path)


def write_error_analysis_csv(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "user_id",
        "claim_object",
        "row_exact_match",
        "mismatched_fields",
        "mismatch_detail",
        "claim_extraction_confidence",
        "image_ids",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["per_row_results"]:
            writer.writerow(
                {
                    "row_index": row["row_index"],
                    "user_id": row["user_id"],
                    "claim_object": row["claim_object"],
                    "row_exact_match": row["row_exact_match"],
                    "mismatched_fields": ";".join(row["mismatches"].keys()) if row["mismatches"] else "none",
                    "mismatch_detail": json.dumps(row["mismatches"], ensure_ascii=False) if row["mismatches"] else "",
                    "claim_extraction_confidence": row["claim_extraction_confidence"],
                    "image_ids": ";".join(row["image_ids"]),
                }
            )
    logger.info("Wrote error analysis to %s", path)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_evaluation_report_md(
    *,
    report: dict,
    injection_report: dict,
    usage: UsageTracker,
    cost_estimate: dict,
    projected_test_cost: dict,
    model: str,
    concurrency: int,
    test_claims_count: int,
    test_images_count: int,
    path: str | Path,
) -> None:
    d = usage.as_dict()
    lines: list[str] = []
    lines.append("# Evaluation report")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `evaluation/run_eval.py`.")
    lines.append("")
    lines.append(f"Evaluated against {report['n_rows']} labeled rows in `dataset/sample_claims.csv`.")
    lines.append("")

    lines.append("## Accuracy")
    lines.append("")
    lines.append(f"- **Row-level exact match rate**: {_fmt_pct(report['row_exact_match_rate'])} "
                  f"(every scored field correct on the row)")
    lines.append("")
    lines.append("| Field | Accuracy |")
    lines.append("|---|---|")
    for field, acc in report["field_accuracy"].items():
        lines.append(f"| `{field}` | {_fmt_pct(acc)} |")
    for field in SET_FIELDS:
        exact = report["field_set_exact_match_rate"][field]
        jacc = report["field_set_mean_jaccard"][field]
        lines.append(f"| `{field}` (exact set match) | {_fmt_pct(exact)} |")
        lines.append(f"| `{field}` (mean Jaccard overlap) | {_fmt_pct(jacc)} |")
    lines.append("")

    lines.append("## `claim_status` confusion matrix")
    lines.append("")
    lines.append("Format: `expected -> predicted: count`. A non-empty off-diagonal entry is a")
    lines.append("misclassification; whether errors skew \"too trusting\" (expected contradicted/")
    lines.append("not_enough_information -> predicted supported) or \"too suspicious\" (the reverse)")
    lines.append("is itself a diagnostic signal — see ARCHITECTURE_AND_STRATEGY.md §9.")
    lines.append("")
    for k, v in sorted(report["claim_status_confusion"].items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## Prompt-injection regression check")
    lines.append("")
    lines.append(
        "Independent of sample_claims.csv's labels: every row whose raw transcript trips the "
        "deterministic heuristic scanner (`src.utils.detect_injection_heuristics`) must be "
        "reflected in the pipeline's own `injection_attempt_detected` risk flag. This check "
        "carries over unchanged to `dataset/claims.csv`, since it never references a specific "
        "row's identity or label."
    )
    lines.append("")
    lines.append(f"- Rows with detected injection-shaped language: {injection_report['rows_with_injection_language']}")
    lines.append(f"- Result: {'PASS' if injection_report['passed'] else 'FAIL'}")
    if not injection_report["passed"]:
        lines.append("")
        lines.append("Rows where the pipeline did not flag detected injection language:")
        for failure in injection_report["rows_where_pipeline_missed_it"]:
            lines.append(f"  - `{failure['user_id']}`: detected {failure['detected_phrases']}")
    lines.append("")

    lines.append("## Operational analysis")
    lines.append("")
    lines.append(f"Measurements below are from this evaluation run only (sample set, "
                  f"{d['total_claims']} claims, {d['images_processed']} images), with a linear "
                  f"projection to the full `dataset/claims.csv` test set "
                  f"({test_claims_count} claims, {test_images_count} images, both counted from "
                  f"the input CSV without running the pipeline against it here).")
    lines.append("")
    lines.append("### Model calls")
    lines.append("")
    lines.append(f"- Stage 2 (claim extraction): {d['claim_extraction_calls']} calls "
                  f"({d['claim_extraction_cache_hits']} served from cache)")
    lines.append(f"- Stage 3 (visual assessment): {d['visual_assessment_calls']} calls over "
                  f"{d['images_processed']} images ({d['visual_assessment_cache_hits']} served from cache)")
    lines.append(f"- Stage 4 (decision engine): 0 calls (pure code, by design)")
    lines.append(f"- Projected for the full test set (~{test_claims_count} claims, "
                  f"~{test_images_count} images): ~{test_claims_count} Stage 2 calls + "
                  f"~{test_images_count} Stage 3 calls")
    lines.append("")
    lines.append("### Token usage (measured on the sample set)")
    lines.append("")
    lines.append(f"- Input tokens: {d['input_tokens']}")
    lines.append(f"- Output tokens: {d['output_tokens']}")
    lines.append(f"- Total: {d['total_tokens']}")
    lines.append("")
    lines.append("### Cost")
    lines.append("")
    lines.append("Sample-set measured cost:")
    lines.append(f"```json\n{json.dumps(cost_estimate, indent=2)}\n```")
    lines.append("")
    lines.append("Linear projection to the full test set (see assumption in the JSON):")
    lines.append(f"```json\n{json.dumps(projected_test_cost, indent=2)}\n```")
    lines.append("")
    lines.append("### Latency")
    lines.append("")
    lines.append(f"- Stage 2 wall clock: {d['claim_extraction_seconds']}s for {d['claim_extraction_calls']} calls "
                  f"at concurrency={concurrency}")
    lines.append(f"- Stage 3 wall clock: {d['visual_assessment_seconds']}s for {d['visual_assessment_calls']} calls "
                  f"at concurrency={concurrency}")
    lines.append(f"- Total wall clock for this evaluation run: {d['total_seconds']}s")
    lines.append(
        f"- Naive linear scaling suggests roughly "
        f"{round(d['claim_extraction_seconds'] / max(d['total_claims'], 1) * test_claims_count + d['visual_assessment_seconds'] / max(d['images_processed'], 1) * test_images_count, 1)}s "
        f"wall clock for the full test set at the same concurrency, ignoring queueing effects from "
        f"larger batches."
    )
    lines.append("")
    lines.append("### Rate limits, batching, caching, retries")
    lines.append("")
    lines.append(f"- Concurrency cap used for this run: {concurrency} (set via `--concurrency`, "
                  f"or `MAX_CONCURRENCY` env var; see `src/utils.DEFAULT_CONCURRENCY`).")
    lines.append("- Stage 2 and Stage 3 are run as two separate bounded-concurrency batches "
                  "(all claim-extraction calls, then all visual-assessment calls) rather than "
                  "interleaved, to keep the in-flight request count predictable against the "
                  "provider's TPM/RPM limits.")
    lines.append("- Transient errors (HTTP 429, 5xx, timeouts) are retried with exponential "
                  "backoff and jitter (`src.utils._generate_content_with_backoff`, up to 5 "
                  "attempts).")
    lines.append("- Schema-validation failures get one reprompt with the validation error "
                  "appended before falling back to a conservative default "
                  "(`evidence_standard_met=false` / `not_enough_information`) rather than "
                  "crashing the row.")
    lines.append("- Both LLM stages use a disk-backed, content-hash-keyed cache "
                  "(`src.utils.JSONCache`): identical image bytes or normalized transcript text "
                  "are never re-sent to the model on a re-run. Cache hit counts above reflect "
                  "this run; a fully warm cache run costs ~0 additional tokens.")
    lines.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote evaluation report to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the pipeline against dataset/sample_claims.csv and write an evaluation report."
    )
    parser.add_argument("--sample-csv", default=str(DEFAULT_SAMPLE_CSV))
    parser.add_argument("--evidence-csv", default=str(DEFAULT_EVIDENCE_CSV))
    parser.add_argument("--history-csv", default=str(DEFAULT_HISTORY_CSV))
    parser.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument(
        "--test-claims-csv",
        default=None,
        help=(
            "Optional path to dataset/claims.csv, used ONLY to count rows/images for the "
            "operational-analysis cost/latency projection in evaluation_report.md. The "
            "pipeline is NOT run against this file by this script — use main.py for that."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_EVAL_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _count_claims_and_images(csv_path: str | Path) -> tuple[int, int]:
    rows = read_csv_rows(csv_path)
    image_count = sum(len([p for p in row["image_paths"].split(";") if p.strip()]) for row in rows)
    return len(rows), image_count


async def _main_async(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    output_dir = Path(args.output_dir)

    started = time.perf_counter()
    traces, usage = await run_pipeline(
        claims_csv=args.sample_csv,
        evidence_csv=args.evidence_csv,
        history_csv=args.history_csv,
        images_root=args.images_root,
        model=args.model,
        concurrency=args.concurrency,
        use_cache=not args.no_cache,
    )
    usage.total_seconds = time.perf_counter() - started

    labels = load_labels(args.sample_csv)
    report = score(traces, labels)
    injection_report = injection_regression_check(traces)

    pricing = load_pricing()
    cost_estimate = estimate_cost(usage, pricing, args.model)

    if args.test_claims_csv:
        test_claims_count, test_images_count = _count_claims_and_images(args.test_claims_csv)
    else:
        # No claims.csv given: fall back to the sample set's own size so the report
        # still renders something meaningful and clearly-scoped rather than crashing.
        test_claims_count, test_images_count = usage.total_claims, usage.images_processed
        logger.warning(
            "--test-claims-csv not provided; projecting cost/latency using the sample "
            "set's own size (%d claims, %d images) instead of dataset/claims.csv.",
            test_claims_count,
            test_images_count,
        )

    projected_test_cost = project_full_test_cost(
        usage, usage.total_claims, test_claims_count, pricing, args.model
    )

    write_sample_predictions_csv(traces, output_dir / "sample_predictions.csv")
    write_error_analysis_csv(report, output_dir / "error_analysis.csv")
    write_evaluation_report_md(
        report=report,
        injection_report=injection_report,
        usage=usage,
        cost_estimate=cost_estimate,
        projected_test_cost=projected_test_cost,
        model=args.model,
        concurrency=args.concurrency,
        test_claims_count=test_claims_count,
        test_images_count=test_images_count,
        path=output_dir / "evaluation_report.md",
    )

    logger.info(
        "Evaluation complete: row_exact_match_rate=%.1f%% | injection_check=%s | "
        "see %s/evaluation_report.md",
        report["row_exact_match_rate"] * 100,
        "PASS" if injection_report["passed"] else "FAIL",
        output_dir,
    )

    # Non-zero exit code if the injection regression check fails, so this can be
    # wired into a CI step / pre-submission check without parsing the markdown.
    if not injection_report["passed"]:
        sys.exit(1)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()