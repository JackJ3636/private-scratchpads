"""Produce the final analysis set by joining correctness grades and illegibility assessments.

Reads parsed traces, correctness grades, and illegibility assessments,
then filters to traces that are correct (or partially correct) AND
sufficiently illegible. Outputs analysis_ids.json and a merged JSONL.

Usage:
    python src/filter.py \
        --traces runs/parsed_qwq.jsonl \
        --grades runs/graded_qwq.jsonl \
        --assessments runs/assessed_qwq.jsonl \
        --output runs/analysis_ids.json \
        --merged runs/analysis_set.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter to analysis set")
    parser.add_argument("--traces", type=str, required=True, help="Parsed traces JSONL")
    parser.add_argument("--grades", type=str, required=True, help="Correctness grades JSONL")
    parser.add_argument("--assessments", type=str, required=True, help="Illegibility assessments JSONL")
    parser.add_argument("--output", type=str, default="runs/analysis_ids.json", help="Output analysis IDs JSON")
    parser.add_argument("--merged", type=str, default="runs/analysis_set.jsonl", help="Output merged analysis set JSONL")
    parser.add_argument("--min-illegibility", type=int, default=5, help="Minimum illegibility score (1-10)")
    parser.add_argument("--correctness", nargs="+", default=["correct", "partially_correct"],
                        help="Accepted correctness labels")
    parser.add_argument("--min-cot-length", type=int, default=100, help="Minimum CoT length in chars")
    args = parser.parse_args()

    # Load all three data sources, keyed by ID
    traces = {}
    with open(args.traces, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            traces[r["id"]] = r

    grades = {}
    with open(args.grades, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            grades[r["id"]] = r

    assessments = {}
    with open(args.assessments, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            assessments[r["id"]] = r

    print(f"Loaded: {len(traces)} traces, {len(grades)} grades, {len(assessments)} assessments")

    # Filter
    analysis_ids = []
    rejected = {"no_grade": 0, "no_assessment": 0, "incorrect": 0,
                "not_illegible": 0, "short_cot": 0, "no_answer": 0}

    for trace_id, trace in traces.items():
        if trace_id not in grades:
            rejected["no_grade"] += 1
            continue
        if trace_id not in assessments:
            rejected["no_assessment"] += 1
            continue

        grade = grades[trace_id]
        assessment = assessments[trace_id]

        # Must have a final answer
        if not trace.get("final_answer"):
            rejected["no_answer"] += 1
            continue

        # Correctness check
        if grade.get("correctness") not in args.correctness:
            rejected["incorrect"] += 1
            continue

        # Illegibility check
        score = assessment.get("assessment", {}).get("score", 0)
        if score < args.min_illegibility:
            rejected["not_illegible"] += 1
            continue

        # CoT length check
        cot_len = len(trace.get("cot_text", ""))
        if cot_len < args.min_cot_length:
            rejected["short_cot"] += 1
            continue

        analysis_ids.append(trace_id)

    print(f"\n--- Filtering Results ---")
    print(f"Analysis set: {len(analysis_ids)} traces")
    print(f"Rejected:")
    for reason, count in rejected.items():
        if count > 0:
            print(f"  {reason}: {count}")

    # Save analysis IDs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "analysis_ids": sorted(analysis_ids),
            "n": len(analysis_ids),
            "criteria": {
                "correctness": args.correctness,
                "min_illegibility": args.min_illegibility,
                "min_cot_length": args.min_cot_length,
            },
        }, f, indent=2)
    print(f"\nSaved {len(analysis_ids)} IDs to {output_path}")

    # Save merged analysis set (traces with grades + assessments attached)
    merged_path = Path(args.merged)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        for trace_id in sorted(analysis_ids):
            trace = traces[trace_id]
            record = {
                "id": trace_id,
                "dataset": trace["dataset"],
                "model": trace["model"],
                "question": trace["question"],
                "gold_answer": trace["gold_answer"],
                "cot_text": trace["cot_text"],
                "final_answer": trace["final_answer"],
                "heuristics": trace["heuristics"],
                "correctness": grades[trace_id]["correctness"],
                "illegibility_score": assessments[trace_id]["assessment"]["score"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved merged analysis set to {merged_path}")

    # Quick stats on the analysis set
    illeg_scores = [assessments[id]["assessment"]["score"] for id in analysis_ids]
    if illeg_scores:
        print(f"\nAnalysis set stats:")
        print(f"  Mean illegibility: {sum(illeg_scores)/len(illeg_scores):.1f}/10")
        correct_only = sum(1 for id in analysis_ids if grades[id]["correctness"] == "correct")
        partial_only = sum(1 for id in analysis_ids if grades[id]["correctness"] == "partially_correct")
        print(f"  Correct: {correct_only}, Partially correct: {partial_only}")


if __name__ == "__main__":
    main()
