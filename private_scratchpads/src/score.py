"""Score decoded reader outputs and produce the primary results table.

Two-stage scoring:
  1. GPT-4o grades each (reader_output, gold_answer) pair for correctness
  2. Compute per-condition accuracy with bootstrap CIs + McNemar's tests

Usage:
    # Stage 1: Grade reader outputs
    python src/score.py grade \
        --decoded runs/decoded.jsonl \
        --output runs/scored.jsonl \
        --workers 5

    # Stage 2: Compute results table (no API calls)
    python src/score.py results \
        --scored runs/scored.jsonl \
        --output-dir results/tables
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import concurrent.futures

import numpy as np
from openai import OpenAI


# --- Stage 1: GPT-4o grading ---

GRADING_PROMPT = """\
Compare the predicted answer with the actual answer and determine if the prediction is:
- "correct": The prediction correctly addresses the question and aligns with the actual answer
- "partially_correct": The prediction has some correct elements but also contains errors
- "incorrect": The prediction is wrong or completely misaligns with the actual answer

Consider that answers might be worded differently but still convey the same meaning.
For numerical answers, allow minor rounding differences.

QUESTION: {question}

PREDICTED ANSWER: {predicted}

ACTUAL ANSWER: {gold}

Respond with JSON: {{"correctness": "correct|partially_correct|incorrect", "explanation": "<brief explanation>"}}"""


def record_key(r):
    """Unique key for a record — works for both decode and scrub_decode formats."""
    cond = r.get("condition") or r.get("scrub", "unknown")
    model = r.get("reader_model", "")
    return (r["trace_id"], cond, r["reader"], model)


def record_label(r):
    """Short label for progress logging."""
    return r.get("condition") or r.get("scrub", "?")


def grade_one(record, model, client):
    """Grade a single decoded record."""
    predicted = record["reader_output"]
    if predicted.startswith("ERROR:"):
        return {**record, "correctness": "error", "explanation": "Reader call failed"}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that grades answer correctness. Always respond with valid JSON."},
                {"role": "user", "content": GRADING_PROMPT.format(
                    question=record.get("question", ""),
                    predicted=predicted[:3000],
                    gold=record["gold_answer"],
                )},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return {**record, **result}
    except Exception as e:
        return {**record, "correctness": "error", "explanation": str(e)}


def run_grading(args):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    records = []
    with open(args.decoded, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} decoded records")

    # Resume
    done_keys = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("correctness") != "error":
                    done_keys.add(record_key(r))
        print(f"Resuming: {len(done_keys)} already scored (skipping errors)")

    remaining = [r for r in records if record_key(r) not in done_keys]
    if args.limit:
        remaining = remaining[:args.limit]
    print(f"Grading {len(remaining)} records with {args.model}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_rec = {
            executor.submit(grade_one, r, args.model, client): r
            for r in remaining
        }
        with open(output_path, "a", encoding="utf-8") as fout:
            for future in concurrent.futures.as_completed(future_to_rec):
                try:
                    result = future.result()
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    completed += 1
                    cond = record_label(result)
                    reader = result["reader"]
                    corr = result.get("correctness", "?")
                    if completed % 20 == 0 or completed == len(remaining):
                        print(f"Progress: {completed}/{len(remaining)}")
                except Exception as e:
                    rec = future_to_rec[future]
                    print(f"Error: {rec['trace_id']}/{record_label(rec)}: {e}")

    print(f"\nGrading complete. {completed} records written to {output_path}")


# --- Stage 2: Results computation ---

def bootstrap_ci(correct, n_resamples=10000, ci=0.95, seed=42):
    """Compute accuracy with bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    arr = np.array(correct, dtype=float)
    acc = arr.mean()

    boot_accs = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_accs.append(sample.mean())

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_accs, 100 * alpha)
    ci_upper = np.percentile(boot_accs, 100 * (1 - alpha))
    return float(acc), float(ci_lower), float(ci_upper)


def mcnemar_test(correct_a, correct_b):
    """McNemar's test for paired binary outcomes."""
    assert len(correct_a) == len(correct_b)

    # b = A correct, B wrong; c = A wrong, B correct
    b = sum(a and not bb for a, bb in zip(correct_a, correct_b))
    c = sum(not a and bb for a, bb in zip(correct_a, correct_b))

    n = b + c
    if n == 0:
        return {"statistic": 0, "p_value": 1.0, "b": b, "c": c}

    from scipy.stats import binomtest
    result = binomtest(b, n, 0.5)
    stat = (b - c) ** 2 / n if n > 0 else 0

    return {"statistic": stat, "p_value": result.pvalue, "b": b, "c": c}


def run_results(args):
    records = []
    with open(args.scored, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} scored records\n")

    # Group by condition+reader
    groups = defaultdict(list)
    for r in records:
        key = (r["condition"], r["reader"])
        is_correct = r.get("correctness") == "correct"
        groups[key].append({"trace_id": r["trace_id"], "correct": is_correct, "correctness": r.get("correctness")})

    # Print results table
    print("=" * 80)
    print("PRIMARY RESULTS TABLE")
    print("=" * 80)
    print(f"{'Condition':<30} {'N':>5} {'Acc':>7} {'95% CI':>16} {'Correct':>8} {'Partial':>8}")
    print("-" * 80)

    condition_results = {}
    for (cond, reader), entries in sorted(groups.items()):
        correct_list = [e["correct"] for e in entries]
        acc, ci_lo, ci_hi = bootstrap_ci(correct_list)
        n = len(entries)
        n_correct = sum(1 for e in entries if e["correctness"] == "correct")
        n_partial = sum(1 for e in entries if e["correctness"] == "partially_correct")

        label = f"{cond} / {reader}"
        print(f"{label:<30} {n:>5} {acc:>6.1%} [{ci_lo:>5.1%}, {ci_hi:>5.1%}] {n_correct:>8} {n_partial:>8}")

        condition_results[(cond, reader)] = {
            "correct_list": correct_list,
            "accuracy": acc,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n": n,
            "trace_ids": [e["trace_id"] for e in entries],
        }

    # Paired comparisons (McNemar's tests)
    print("\n" + "=" * 80)
    print("PAIRED COMPARISONS (McNemar's test)")
    print("=" * 80)

    comparisons = [
        ("C1", "self", "C3", "cross", "Self vs Cross (no question)"),
        ("C4", "self", "C5", "cross", "Self vs Cross (with question)"),
        ("C1", "self", "C4", "self", "Self: no-q vs with-q"),
        ("C3", "cross", "C5", "cross", "Cross: no-q vs with-q"),
    ]

    for cond_a, reader_a, cond_b, reader_b, label in comparisons:
        key_a = (cond_a, reader_a)
        key_b = (cond_b, reader_b)
        if key_a not in condition_results or key_b not in condition_results:
            continue

        res_a = condition_results[key_a]
        res_b = condition_results[key_b]

        # Align by trace_id for paired test
        ids_a = {tid: c for tid, c in zip(res_a["trace_ids"], res_a["correct_list"])}
        ids_b = {tid: c for tid, c in zip(res_b["trace_ids"], res_b["correct_list"])}
        shared_ids = sorted(set(ids_a.keys()) & set(ids_b.keys()))

        if not shared_ids:
            print(f"  {label}: no overlapping traces")
            continue

        paired_a = [ids_a[tid] for tid in shared_ids]
        paired_b = [ids_b[tid] for tid in shared_ids]

        test = mcnemar_test(paired_a, paired_b)
        delta = res_a["accuracy"] - res_b["accuracy"]
        sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "ns"

        print(f"  {label}")
        print(f"    Δ = {delta:+.1%}  (A={res_a['accuracy']:.1%}, B={res_b['accuracy']:.1%})")
        print(f"    McNemar p={test['p_value']:.4f} {sig}  (b={test['b']}, c={test['c']}, n_paired={len(shared_ids)})")

    # C6 baseline check
    print("\n" + "=" * 80)
    print("CONTROL: C6 (Random CoT)")
    print("=" * 80)
    for reader in ["self", "cross"]:
        key = ("C6", reader)
        if key in condition_results:
            res = condition_results[key]
            print(f"  {reader}: {res['accuracy']:.1%} [{res['ci_lower']:.1%}, {res['ci_upper']:.1%}] (N={res['n']})")
            print(f"    Expected: near 0% (random CoT should not help)")

    # Save JSON results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = {}
    for (cond, reader), res in condition_results.items():
        results_json[f"{cond}_{reader}"] = {
            "accuracy": res["accuracy"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
            "n": res["n"],
        }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results to {output_dir / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Score decoded results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Grade subcommand
    grade_parser = subparsers.add_parser("grade", help="Grade reader outputs with GPT-4o")
    grade_parser.add_argument("--decoded", type=str, required=True, help="Decoded JSONL")
    grade_parser.add_argument("--output", type=str, default="runs/scored.jsonl", help="Output scored JSONL")
    grade_parser.add_argument("--model", type=str, default="gpt-4o")
    grade_parser.add_argument("--workers", type=int, default=5)
    grade_parser.add_argument("--limit", type=int, default=None)
    grade_parser.add_argument("--resume", action="store_true")

    # Results subcommand
    results_parser = subparsers.add_parser("results", help="Compute results table from scored data")
    results_parser.add_argument("--scored", type=str, default="runs/scored.jsonl")
    results_parser.add_argument("--output-dir", type=str, default="results/tables")

    args = parser.parse_args()

    if args.command == "grade":
        run_grading(args)
    elif args.command == "results":
        run_results(args)


if __name__ == "__main__":
    main()
