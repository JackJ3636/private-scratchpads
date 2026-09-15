"""Grade correctness of model answers against gold answers using GPT-4o.

For free-text datasets like GPQA where exact-match doesn't work.

Usage:
    python src/grade_correctness.py \
        --input runs/parsed_qwq.jsonl \
        --output runs/graded_qwq.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path
import concurrent.futures

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CORRECTNESS_PROMPT = """\
Compare the predicted answer with the actual answer and determine if the prediction is:
- "correct": The prediction correctly addresses the question and aligns with the actual answer
- "partially_correct": The prediction has some correct elements but also contains errors or misunderstandings
- "incorrect": The prediction is wrong or completely misaligns with the actual answer

Consider that answers might be worded differently but still convey the same meaning.
For multiple-choice questions, focus on whether the same option was selected.

QUESTION: {question}

PREDICTED ANSWER: {predicted}

ACTUAL ANSWER: {gold}

Respond with JSON: {{"correctness": "correct|partially_correct|incorrect", "explanation": "<brief explanation>"}}"""


def grade_one(trace, model, client):
    """Grade a single trace's correctness."""
    trace_id = trace["id"]
    predicted = trace.get("final_answer")
    gold = trace.get("gold_answer", "")
    question = trace.get("question", "")

    if not predicted:
        return {
            "id": trace_id,
            "correctness": "no_answer",
            "explanation": "No final answer extracted from model output",
            "timestamp": time.time(),
        }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that grades answer correctness. Always respond with valid JSON."},
                {"role": "user", "content": CORRECTNESS_PROMPT.format(
                    question=question,
                    predicted=predicted[:3000],
                    gold=gold,
                )},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        print(f"  {trace_id}: {result['correctness']}")
        return {
            "id": trace_id,
            **result,
            "timestamp": time.time(),
        }
    except Exception as e:
        print(f"  {trace_id}: ERROR - {e}")
        return {
            "id": trace_id,
            "correctness": "error",
            "explanation": str(e),
            "timestamp": time.time(),
        }


def main():
    parser = argparse.ArgumentParser(description="Grade correctness with GPT-4o")
    parser.add_argument("--input", type=str, required=True, help="Parsed traces JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output grades JSONL")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Grading model")
    parser.add_argument("--limit", type=int, default=None, help="Limit traces to process")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent workers")
    parser.add_argument("--resume", action="store_true", help="Skip already-graded traces")
    args = parser.parse_args()

    # Load parsed traces
    traces = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))
    print(f"Loaded {len(traces)} traces from {args.input}")

    # Resume support
    done_ids = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} already graded")

    remaining = [t for t in traces if t["id"] not in done_ids]
    if args.limit:
        remaining = remaining[:args.limit]
    print(f"Grading {len(remaining)} traces with {args.model}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_trace = {
            executor.submit(grade_one, t, args.model, client): t
            for t in remaining
        }
        with open(output_path, "a", encoding="utf-8") as fout:
            for future in concurrent.futures.as_completed(future_to_trace):
                try:
                    record = future.result()
                    results.append(record)
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    print(f"Progress: {len(results)}/{len(remaining)}")
                except Exception as e:
                    trace = future_to_trace[future]
                    print(f"Error processing {trace['id']}: {e}")

    # Summary
    print("\n--- Summary ---")
    all_grades = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            all_grades.append(json.loads(line))

    from collections import Counter
    counts = Counter(g["correctness"] for g in all_grades)
    total = len(all_grades)
    print(f"Total graded: {total}")
    for label in ["correct", "partially_correct", "incorrect", "no_answer", "error"]:
        n = counts.get(label, 0)
        pct = n / total * 100 if total else 0
        print(f"  {label}: {n} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
