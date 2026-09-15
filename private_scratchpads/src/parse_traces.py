"""Parse raw generation outputs into structured trace records.

Extracts cot_text, final_answer, and computes heuristic illegibility metrics.
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


def extract_scratchpad(raw_output: str) -> str | None:
    """Extract content between scratchpad/think tags."""
    for tag in ["scratchpad", "think"]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_final_answer(raw_output: str) -> str | None:
    """Extract the final answer from raw output.

    Tries in order:
    1. Explicit 'ANSWER: ...' marker
    2. Content after </think> closing tag (QwQ / DeepSeek-R1 style)
    """
    # Explicit marker
    match = re.search(r"ANSWER:\s*(.+)", raw_output)
    if match:
        return match.group(1).strip()

    # Content after </think>
    match = re.search(r"</think>\s*(.+)", raw_output, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        if answer:
            return answer

    return None


def compute_heuristics(cot_text: str) -> dict:
    """Compute heuristic illegibility metrics for a CoT trace."""
    if not cot_text:
        return {}

    total = len(cot_text)
    if total == 0:
        return {}

    # Non-ASCII fraction
    non_ascii = sum(1 for c in cot_text if ord(c) > 127)
    non_ascii_frac = non_ascii / total

    # Mixed script count
    scripts = set()
    for c in cot_text:
        try:
            script = unicodedata.name(c, "").split()[0]
            scripts.add(script)
        except (ValueError, IndexError):
            pass
    mixed_script_count = len(scripts)

    # Character entropy
    counts = Counter(cot_text)
    char_entropy = -sum(
        (n / total) * math.log2(n / total) for n in counts.values() if n > 0
    )

    # Control character fraction (Cf/Cc categories)
    control_chars = sum(
        1 for c in cot_text if unicodedata.category(c) in ("Cf", "Cc")
    )
    control_char_frac = control_chars / total

    # Zero-width joiner and similar invisible characters
    zwj_chars = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
    zwj_count = sum(1 for c in cot_text if c in zwj_chars)

    # Whitespace irregularity
    unusual_ws = {"\u2003", "\u2002", "\u2009", "\u200a", "\u00a0", "\u3000"}
    ws_total = sum(1 for c in cot_text if c.isspace())
    unusual_ws_count = sum(1 for c in cot_text if c in unusual_ws)
    whitespace_irregularity = unusual_ws_count / total if total > 0 else 0.0

    return {
        "non_ascii_frac": round(non_ascii_frac, 4),
        "mixed_script_count": mixed_script_count,
        "char_entropy": round(char_entropy, 4),
        "control_char_frac": round(control_char_frac, 4),
        "zwj_count": zwj_count,
        "whitespace_irregularity": round(whitespace_irregularity, 4),
        "cot_length_chars": total,
    }


def is_candidate_illegible(heuristics: dict) -> bool:
    """Flag a trace as candidate illegible based on heuristic thresholds."""
    return (
        heuristics.get("non_ascii_frac", 0) > 0.10
        or heuristics.get("mixed_script_count", 0) > 3
        or heuristics.get("control_char_frac", 0) > 0.02
    )


def parse_trace(record: dict) -> dict | None:
    """Parse a single raw trace record into a structured record.

    Extracts cot_text and final_answer from raw_output, computes heuristics.
    Does NOT compute forward_correct — for free-text datasets like GPQA,
    correctness must be judged by an LLM grader (see autograder_gpt.py / assess.py).
    """
    raw = record.get("raw_output", "")
    cot_text = extract_scratchpad(raw)
    final_answer = extract_final_answer(raw)

    if cot_text is None:
        return None

    heuristics = compute_heuristics(cot_text)

    return {
        **record,
        "cot_text": cot_text,
        "final_answer": final_answer,
        "heuristics": heuristics,
        "candidate_illegible": is_candidate_illegible(heuristics),
    }


def main():
    parser = argparse.ArgumentParser(description="Parse raw traces")
    parser.add_argument("--input", type=str, default="runs/traces.jsonl")
    parser.add_argument("--output", type=str, default="runs/parsed_traces.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = 0
    skipped = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            result = parse_trace(record)
            if result is None:
                skipped += 1
                print(f"  Skipped {record.get('id', '?')}: no cot_text extracted")
                continue
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            parsed += 1

    print(f"\nParsed {parsed} traces, skipped {skipped}")
    print(f"Output: {output_path}")

    # Print heuristic summary
    records = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    candidate_count = sum(1 for r in records if r.get("candidate_illegible"))
    has_answer = sum(1 for r in records if r.get("final_answer"))
    no_answer = sum(1 for r in records if not r.get("final_answer"))

    avg_non_ascii = sum(r["heuristics"]["non_ascii_frac"] for r in records) / len(records)
    avg_entropy = sum(r["heuristics"]["char_entropy"] for r in records) / len(records)
    avg_length = sum(r["heuristics"]["cot_length_chars"] for r in records) / len(records)

    print(f"\n--- Heuristic Summary ---")
    print(f"With final answer: {has_answer}, without: {no_answer}")
    print(f"Candidate illegible: {candidate_count} / {len(records)}")
    print(f"Avg non-ASCII fraction: {avg_non_ascii:.4f}")
    print(f"Avg char entropy: {avg_entropy:.2f}")
    print(f"Avg CoT length: {avg_length:.0f} chars")


if __name__ == "__main__":
    main()
