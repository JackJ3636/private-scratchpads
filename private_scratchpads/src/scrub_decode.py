"""Phase 3: Scrubbing experiments.

Applies each scrub (S0-S7) to the analysis set CoTs, then runs reader
decoding on the scrubbed versions through self and cross readers.

Usage:
    python src/scrub_decode.py \
        --analysis-set runs/analysis_set.jsonl \
        --scrubs S0 S1 S2 S3 S4 S5 S6 S7 \
        --output runs/scrub_decoded.jsonl \
        --workers 5
"""

import argparse
import json
import os
import time
from pathlib import Path
import concurrent.futures

from openai import OpenAI
from scrub import apply_scrub

PROMPT_NO_Q = (Path(__file__).parent.parent / "prompts" / "reader_no_question.txt").read_text()

# --- Clients ---

def make_client(api):
    configs = {
        "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1"},
        "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": None},
    }
    cfg = configs[api]
    kwargs = {"api_key": os.getenv(cfg["api_key_env"])}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


READER_PARAMS = {"temperature": 0.0, "max_tokens": 200}


def call_reader(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **READER_PARAMS,
    )
    return response.choices[0].message.content.strip()


def process_one(trace, scrub_id, reader_label, client, model):
    """Apply scrub and run reader on one trace."""
    trace_id = trace["id"]
    cot = trace["cot_text"]

    # Apply scrub
    scrubbed = apply_scrub(cot, scrub_id)

    # Build prompt (no question — matches C1/C3 for clean comparison)
    prompt = PROMPT_NO_Q.replace("{cot_text}", scrubbed)

    try:
        output = call_reader(client, model, prompt)
    except Exception as e:
        output = f"ERROR: {e}"

    return {
        "trace_id": trace_id,
        "scrub": scrub_id,
        "reader": reader_label,
        "reader_model": model,
        "reader_output": output,
        "gold_answer": trace["gold_answer"],
        "question": trace["question"],
        "scrubbed_length": len(scrubbed),
        "original_length": len(cot),
        "timestamp": time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run scrubbing experiments (Phase 3)")
    parser.add_argument("--analysis-set", type=str, required=True)
    parser.add_argument("--scrubs", nargs="+", default=["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"])
    parser.add_argument("--self-model", type=str, default="qwen/qwq-32b")
    parser.add_argument("--self-api", type=str, default="openrouter")
    parser.add_argument("--cross-model", type=str, default="gpt-4o")
    parser.add_argument("--cross-api", type=str, default="openai")
    parser.add_argument("--readers", nargs="+", default=["self", "cross"],
                        help="Which readers to run (self, cross, or both)")
    parser.add_argument("--output", type=str, default="runs/scrub_decoded.jsonl")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load traces
    traces = []
    with open(args.analysis_set, encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))
    if args.limit:
        traces = traces[:args.limit]
    print(f"Loaded {len(traces)} traces")
    print(f"Scrubs: {args.scrubs}")
    print(f"Readers: {args.readers}")

    # Resume
    done_keys = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r["reader_output"].startswith("ERROR:"):
                    done_keys.add((r["trace_id"], r["scrub"], r["reader_model"]))
        print(f"Resuming: {len(done_keys)} already done (non-error)")

    # Build tasks
    reader_configs = []
    if "self" in args.readers:
        reader_configs.append(("self", make_client(args.self_api), args.self_model))
    if "cross" in args.readers:
        reader_configs.append(("cross", make_client(args.cross_api), args.cross_model))

    tasks = []
    for scrub_id in args.scrubs:
        for trace in traces:
            for reader_label, client, model in reader_configs:
                if (trace["id"], scrub_id, model) in done_keys:
                    continue
                tasks.append((trace, scrub_id, reader_label, client, model))

    total = len(tasks)
    print(f"\nRunning {total} tasks ({len(args.scrubs)} scrubs x {len(traces)} traces x {len(reader_configs)} readers)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(process_one, t, s, rl, c, m): (t, s, rl)
            for t, s, rl, c, m in tasks
        }

        with open(output_path, "a", encoding="utf-8") as fout:
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    record = future.result()
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    completed += 1
                    if completed % 20 == 0 or completed == total:
                        print(f"Progress: {completed}/{total}")
                except Exception as e:
                    task_info = future_to_task[future]
                    print(f"Error: {task_info[0]['id']}/{task_info[1]}/{task_info[2]}: {e}")

    # Summary
    print("\n--- Summary ---")
    all_records = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            all_records.append(json.loads(line))

    from collections import Counter
    by_key = Counter((r["scrub"], r["reader"]) for r in all_records)
    for (scrub, reader), count in sorted(by_key.items()):
        err = sum(1 for r in all_records if r["scrub"] == scrub and r["reader"] == reader
                  and r["reader_output"].startswith("ERROR:"))
        print(f"  {scrub} / {reader}: {count} ({err} errors)")


if __name__ == "__main__":
    main()
