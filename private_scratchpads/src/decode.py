"""Phase 2: Reader decoding runs.

Runs self-reader and cross-reader models on CoT traces under
various conditions and saves results.

Models are configurable via CLI args. Both self and cross readers
use the OpenAI-compatible API (works with OpenRouter, OpenAI, etc).

Usage:
    # Main experiment (QwQ self-reader, GPT-4o cross-reader):
    python src/decode.py \
        --analysis-set runs/analysis_set.jsonl \
        --conditions C1 C3 C4 C5 C6 \
        --self-model qwen/qwq-32b --self-api openrouter \
        --cross-model gpt-4o --cross-api openai \
        --output runs/decoded.jsonl --workers 5

    # Same-family cross-reader (C2):
    python src/decode.py \
        --analysis-set runs/analysis_set.jsonl \
        --conditions C2 \
        --cross-model qwen/qwen-2.5-32b-instruct --cross-api openrouter \
        --output runs/decoded_qwen_cross.jsonl --workers 5

    # Weak cross-reader:
    python src/decode.py \
        --analysis-set runs/analysis_set.jsonl \
        --conditions C3 C5 \
        --cross-model anthropic/claude-3.5-haiku --cross-api openrouter \
        --output runs/decoded_haiku.jsonl --workers 5

    # C7 (legible control, self-reader):
    python src/decode.py \
        --analysis-set runs/analysis_set.jsonl \
        --legible-traces runs/legible_set.jsonl \
        --conditions C7 \
        --self-model qwen/qwq-32b --self-api openrouter \
        --output runs/decoded_c7.jsonl --workers 5
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
import concurrent.futures

from openai import OpenAI

PROMPT_NO_Q = (Path(__file__).parent.parent / "prompts" / "reader_no_question.txt").read_text()
PROMPT_WITH_Q = (Path(__file__).parent.parent / "prompts" / "reader_with_question.txt").read_text()

# --- Clients ---

API_CONFIGS = {
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,  # default
    },
}

def make_client(api: str) -> OpenAI:
    cfg = API_CONFIGS[api]
    kwargs = {"api_key": os.getenv(cfg["api_key_env"])}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)

# --- Condition definitions ---

CONDITIONS = {
    "C1": {"description": "Self-reader, CoT only",             "reader": "self",  "include_question": False},
    "C2": {"description": "Cross-same-family, CoT only",       "reader": "cross", "include_question": False},
    "C3": {"description": "Cross-reader, CoT only",            "reader": "cross", "include_question": False},
    "C4": {"description": "Self-reader, with question",        "reader": "self",  "include_question": True},
    "C5": {"description": "Cross-reader, with question",       "reader": "cross", "include_question": True},
    "C6": {"description": "Random CoT control",                "reader": "both",  "include_question": False, "random_cot": True},
    "C7": {"description": "Legible CoT (self or cross)",       "reader": "self",  "include_question": False, "legible_only": True},
}

READER_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 200,
}


def build_prompt(cot_text: str, question: str | None = None) -> str:
    if question:
        return PROMPT_WITH_Q.replace("{question}", question).replace("{cot_text}", cot_text)
    return PROMPT_NO_Q.replace("{cot_text}", cot_text)


def call_reader(client: OpenAI, model: str, prompt: str) -> str:
    """Call a reader model and return the raw response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **READER_PARAMS,
    )
    return response.choices[0].message.content.strip()


def run_one(trace, condition, cond_def, self_client, self_model,
            cross_client, cross_model, cot_override=None):
    """Run a single trace through a condition. Returns list of records."""
    cot = cot_override if cot_override else trace["cot_text"]
    question = trace["question"] if cond_def["include_question"] else None
    prompt = build_prompt(cot, question)

    readers = []
    if cond_def["reader"] == "self":
        readers = [("self", self_client, self_model)]
    elif cond_def["reader"] == "cross":
        readers = [("cross", cross_client, cross_model)]
    elif cond_def["reader"] == "both":
        readers = [("self", self_client, self_model), ("cross", cross_client, cross_model)]

    results = []
    for reader_label, client, model in readers:
        try:
            output = call_reader(client, model, prompt)
        except Exception as e:
            output = f"ERROR: {e}"

        results.append({
            "trace_id": trace["id"],
            "condition": condition,
            "reader": reader_label,
            "reader_model": model,
            "reader_output": output,
            "gold_answer": trace["gold_answer"],
            "question": trace["question"],
            "timestamp": time.time(),
        })

    return results


def process_task(task, self_client, self_model, cross_client, cross_model):
    """Process a single (trace, condition) task."""
    trace, condition, cond_def, cot_override = task
    trace_id = trace["id"]
    reader = cond_def["reader"]
    print(f"  {condition} | {reader} | {trace_id}")

    records = run_one(trace, condition, cond_def,
                      self_client, self_model, cross_client, cross_model,
                      cot_override)

    for r in records:
        preview = r["reader_output"][:80].replace("\n", " ")
        print(f"    -> {r['reader']} ({r['reader_model']}): {preview}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Run reader decoding (Phase 2)")
    parser.add_argument("--analysis-set", type=str, required=True, help="Analysis set JSONL")
    parser.add_argument("--legible-traces", type=str, default=None, help="Legible traces JSONL (for C7)")
    parser.add_argument("--conditions", nargs="+", default=["C1", "C3", "C4", "C5"],
                        help="Conditions to run")
    parser.add_argument("--self-model", type=str, default="qwen/qwq-32b", help="Self-reader model")
    parser.add_argument("--self-api", type=str, default="openrouter", choices=API_CONFIGS.keys(),
                        help="API for self-reader")
    parser.add_argument("--cross-model", type=str, default="gpt-4o", help="Cross-reader model")
    parser.add_argument("--cross-api", type=str, default="openai", choices=API_CONFIGS.keys(),
                        help="API for cross-reader")
    parser.add_argument("--output", type=str, default="runs/decoded.jsonl")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-decoded (trace_id, condition, reader_model) tuples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for C6 shuffling")
    args = parser.parse_args()

    # Load analysis set
    traces = []
    with open(args.analysis_set, encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))
    if args.limit:
        traces = traces[:args.limit]
    print(f"Loaded {len(traces)} analysis traces")
    print(f"Self-reader: {args.self_model} (via {args.self_api})")
    print(f"Cross-reader: {args.cross_model} (via {args.cross_api})")

    # Load legible traces for C7 if needed
    legible_traces = []
    if "C7" in args.conditions:
        if not args.legible_traces:
            print("WARNING: C7 requested but --legible-traces not provided, skipping C7")
            args.conditions = [c for c in args.conditions if c != "C7"]
        else:
            with open(args.legible_traces, encoding="utf-8") as f:
                for line in f:
                    legible_traces.append(json.loads(line))
            print(f"Loaded {len(legible_traces)} legible traces for C7")

    # Resume support — keyed by (trace_id, condition, reader_model) to distinguish different models
    done_keys = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r["reader_output"].startswith("ERROR:"):
                    done_keys.add((r["trace_id"], r["condition"], r["reader_model"]))
        print(f"Resuming: {len(done_keys)} already decoded (non-error)")

    # Build C6 random CoT mapping
    rng = random.Random(args.seed)
    cot_texts = [t["cot_text"] for t in traces]
    shuffled_cots = cot_texts.copy()
    for _ in range(100):
        rng.shuffle(shuffled_cots)
        if all(s != o for s, o in zip(shuffled_cots, cot_texts)):
            break
    c6_map = {traces[i]["id"]: shuffled_cots[i] for i in range(len(traces))}

    # Build task list
    tasks = []
    for condition in args.conditions:
        cond_def = CONDITIONS[condition]

        if condition == "C7":
            task_traces = legible_traces
        else:
            task_traces = traces

        for trace in task_traces:
            cot_override = None
            if cond_def.get("random_cot"):
                cot_override = c6_map.get(trace["id"], trace["cot_text"])

            # Determine which models will be used for this condition
            models_for_cond = []
            if cond_def["reader"] == "self":
                models_for_cond = [(args.self_model,)]
            elif cond_def["reader"] == "cross":
                models_for_cond = [(args.cross_model,)]
            elif cond_def["reader"] == "both":
                models_for_cond = [(args.self_model,), (args.cross_model,)]

            if all((trace["id"], condition, m[0]) in done_keys for m in models_for_cond):
                continue

            tasks.append((trace, condition, cond_def, cot_override))

    print(f"\nRunning {len(tasks)} tasks across conditions: {args.conditions}")

    # Clients
    self_client = make_client(args.self_api)
    cross_client = make_client(args.cross_api)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(process_task, task,
                          self_client, args.self_model,
                          cross_client, args.cross_model): task
            for task in tasks
        }

        with open(output_path, "a", encoding="utf-8") as fout:
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    records = future.result()
                    for record in records:
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    completed += 1
                    print(f"Progress: {completed}/{len(tasks)}")
                except Exception as e:
                    task = future_to_task[future]
                    print(f"Error: {task[0]['id']} / {task[1]}: {e}")

    # Summary
    print("\n--- Summary ---")
    all_records = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            all_records.append(json.loads(line))

    from collections import Counter
    by_key = Counter((r["condition"], r["reader"], r["reader_model"]) for r in all_records)
    for (cond, reader, model), count in sorted(by_key.items()):
        err = sum(1 for r in all_records
                  if r["condition"] == cond and r["reader"] == reader
                  and r["reader_model"] == model and r["reader_output"].startswith("ERROR:"))
        print(f"  {cond} / {reader} / {model}: {count} ({err} errors)")

    errors = sum(1 for r in all_records if r["reader_output"].startswith("ERROR:"))
    print(f"\nTotal records: {len(all_records)}, errors: {errors}")


if __name__ == "__main__":
    main()
