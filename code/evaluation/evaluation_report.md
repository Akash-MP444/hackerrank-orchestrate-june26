# Evaluation report

Generated 2026-06-19T17:21:36+00:00 by `evaluation/run_eval.py`.

Evaluated against 20 labeled rows in `dataset/sample_claims.csv`.

## Accuracy

- **Row-level exact match rate**: 0.0% (every scored field correct on the row)

| Field | Accuracy |
|---|---|
| `claim_status` | 10.0% |
| `evidence_standard_met` | 10.0% |
| `valid_image` | 90.0% |
| `issue_type` | 15.0% |
| `object_part` | 5.0% |
| `severity` | 10.0% |
| `risk_flags` (exact set match) | 5.0% |
| `risk_flags` (mean Jaccard overlap) | 22.7% |
| `supporting_image_ids` (exact set match) | 10.0% |
| `supporting_image_ids` (mean Jaccard overlap) | 10.0% |

## `claim_status` confusion matrix

Format: `expected -> predicted: count`. A non-empty off-diagonal entry is a
misclassification; whether errors skew "too trusting" (expected contradicted/
not_enough_information -> predicted supported) or "too suspicious" (the reverse)
is itself a diagnostic signal — see ARCHITECTURE_AND_STRATEGY.md §9.

- `contradicted->not_enough_information`: 5
- `not_enough_information->not_enough_information`: 2
- `supported->not_enough_information`: 13

## Prompt-injection regression check

Independent of sample_claims.csv's labels: every row whose raw transcript trips the deterministic heuristic scanner (`src.utils.detect_injection_heuristics`) must be reflected in the pipeline's own `injection_attempt_detected` risk flag. This check carries over unchanged to `dataset/claims.csv`, since it never references a specific row's identity or label.

- Rows with detected injection-shaped language: 0
- Result: PASS

## Operational analysis

Measurements below are from this evaluation run only (sample set, 20 claims, 29 images), with a linear projection to the full `dataset/claims.csv` test set (20 claims, 29 images, both counted from the input CSV without running the pipeline against it here).

### Model calls

- Stage 2 (claim extraction): 20 calls (0 served from cache)
- Stage 3 (visual assessment): 29 calls over 29 images (0 served from cache)
- Stage 4 (decision engine): 0 calls (pure code, by design)
- Projected for the full test set (~20 claims, ~29 images): ~20 Stage 2 calls + ~29 Stage 3 calls

### Token usage (measured on the sample set)

- Input tokens: 0
- Output tokens: 0
- Total: 0

### Cost

Sample-set measured cost:
```json
{
  "model": "gemini-2.5-flash",
  "pricing_found": false,
  "note": "No pricing entry for model 'gemini-2.5-flash' found in config/pricing.yaml (expected models.<model>.input_per_million_usd / output_per_million_usd). Reporting token counts only; update pricing.yaml for a dollar estimate.",
  "input_tokens": 0,
  "output_tokens": 0,
  "estimated_cost_usd": null
}
```

Linear projection to the full test set (see assumption in the JSON):
```json
{
  "model": "gemini-2.5-flash",
  "pricing_found": false,
  "note": "No pricing entry for model 'gemini-2.5-flash' found in config/pricing.yaml (expected models.<model>.input_per_million_usd / output_per_million_usd). Reporting token counts only; update pricing.yaml for a dollar estimate.",
  "input_tokens": 0,
  "output_tokens": 0,
  "estimated_cost_usd": null,
  "projected_from_sample_rows": 20,
  "projected_to_test_rows": 20,
  "assumption": "Average tokens-per-claim-row observed on the sample set is assumed to hold for the full test set. Actual cost will vary with conversation length, image count per claim, and image resolution."
}
```

### Latency

- Stage 2 wall clock: 12.39s for 20 calls at concurrency=1
- Stage 3 wall clock: 0.86s for 29 calls at concurrency=1
- Total wall clock for this evaluation run: 15.91s
- Naive linear scaling suggests roughly 13.2s wall clock for the full test set at the same concurrency, ignoring queueing effects from larger batches.

### Rate limits, batching, caching, retries

- Concurrency cap used for this run: 1 (set via `--concurrency`, or `MAX_CONCURRENCY` env var; see `src/utils.DEFAULT_CONCURRENCY`).
- Stage 2 and Stage 3 are run as two separate bounded-concurrency batches (all claim-extraction calls, then all visual-assessment calls) rather than interleaved, to keep the in-flight request count predictable against the provider's TPM/RPM limits.
- Transient errors (HTTP 429, 5xx, timeouts) are retried with exponential backoff and jitter (`src.utils._generate_content_with_backoff`, up to 5 attempts).
- Schema-validation failures get one reprompt with the validation error appended before falling back to a conservative default (`evidence_standard_met=false` / `not_enough_information`) rather than crashing the row.
- Both LLM stages use a disk-backed, content-hash-keyed cache (`src.utils.JSONCache`): identical image bytes or normalized transcript text are never re-sent to the model on a re-run. Cache hit counts above reflect this run; a fully warm cache run costs ~0 additional tokens.
