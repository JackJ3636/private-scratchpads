"""Generate CoT traces from reasoning models — concurrent version.

Like generate.py but uses ThreadPoolExecutor for parallel API calls,
similar to the r1_run.py / r1_zero_run.py scripts.

Usage:
    # DeepSeek R1 via DeepSeek API:
    python src/generate_concurrent.py \
        --model deepseek-reasoner \
        --api-base https://api.deepseek.com/v1 \
        --api-key sk-... \
        --datasets gpqa \
        --n-per-dataset 198 \
        --max-tokens 16384 \
        --raw \
        --workers 10 \
        --output runs/traces_deepseek_r1.jsonl

    # DeepSeek R1 via OpenRouter:
    python src/generate_concurrent.py \
        --model deepseek/deepseek-r1 \
        --api-base https://openrouter.ai/api/v1 \
        --api-key $OPENROUTER_API_KEY \
        --datasets gpqa \
        --n-per-dataset 198 \
        --raw \
        --workers 20 \
        --output runs/traces_r1_openrouter.jsonl

    # QwQ via local vLLM:
    python src/generate_concurrent.py \
        --model Qwen/QwQ-32B \
        --api-base http://localhost:8000/v1 \
        --datasets gpqa \
        --n-per-dataset 198 \
        --raw \
        --workers 8 \
        --output runs/traces_qwq.jsonl
"""

import argparse
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import concurrent.futures
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

load_dotenv()

PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "generator.txt").read_text()

SAMPLING_DEFAULTS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 4096,
}

# How to extract the gold answer from each dataset
DATASET_CONFIGS = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "parse_gold": lambda a: re.search(r"####\s*(.+)", a).group(1).strip(),
    },
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "split": "test",
        "question_field": "question",
        "answer_field": "answerKey",
        "parse_gold": lambda a: a.strip(),
    },
    "math": {
        "hf_path": "hendrycks/competition_math",
        "hf_name": None,
        "split": "test",
        "question_field": "problem",
        "answer_field": "solution",
        "parse_gold": lambda a: (
            m.group(1).strip() if (m := re.search(r"\\boxed\{(.+)\}", a)) else a.strip()
        ),
    },
    "gpqa": {
        "hf_path": "Idavidrein/gpqa",
        "hf_name": "gpqa_diamond",
        "split": "train",  # GPQA only has a train split
        "question_field": "Question",
        "answer_field": "Correct Answer",
        "parse_gold": lambda a: a.strip(),
    },
}

# Thread-safe lock for writing to the output file
_write_lock = threading.Lock()


def load_questions(dataset: str, n: int, seed: int = 42) -> list[dict]:
    """Load and sample n questions from a HuggingFace dataset."""
    cfg = DATASET_CONFIGS[dataset]
    ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])

    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:n]

    questions = []
    for idx in indices:
        row = ds[idx]
        question_text = row[cfg["question_field"]]

        # For ARC, append the choices to the question
        if dataset == "arc_challenge":
            choices = row["choices"]
            choice_lines = []
            for label, text in zip(choices["label"], choices["text"]):
                choice_lines.append(f"  {label}. {text}")
            question_text = question_text + "\n" + "\n".join(choice_lines)

        # For GPQA, use the correct answer text as gold (open-ended, no choices)
        if dataset == "gpqa":
            gold = row["Correct Answer"].strip()

        if dataset != "gpqa":
            gold = cfg["parse_gold"](row[cfg["answer_field"]])

        questions.append({
            "dataset": dataset,
            "id": f"{dataset}_{idx}",
            "question": question_text,
            "gold_answer": gold,
        })

    return questions


def _is_remote_api(base_url) -> bool:
    """Check if the API base URL is a remote service (not local vLLM)."""
    url_str = str(base_url) if base_url else ""
    return any(host in url_str for host in [
        "openrouter.ai", "api.deepseek.com", "dashscope.aliyuncs.com",
        "api.groq.com", "api.together.xyz", "api.deepinfra.com",
    ])


def generate_trace(
    client: OpenAI,
    model: str,
    question: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    raw: bool = False,
) -> str:
    """Send a single question to the API and return the raw output.

    Handles three modes:
      1. raw + remote API  → chat endpoint, extract reasoning from response
      2. raw + local vLLM  → completions endpoint with chat template + <think> prefix
      3. non-raw           → chat endpoint with scratchpad prompt template
    """
    is_remote = _is_remote_api(client.base_url)

    if raw and is_remote:
        # Remote API (OpenRouter, DeepSeek, DashScope, etc.)
        # Reasoning models return thinking in a separate field
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if not response.choices:
            raise RuntimeError(f"Empty response from API: {response}")
        msg = response.choices[0].message

        # Try all known field names for reasoning content:
        #   - reasoning_content (DeepSeek API)
        #   - reasoning (OpenRouter)
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning is None:
            reasoning = getattr(msg, "reasoning", None)

        content = msg.content or ""

        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n{content}"
        else:
            # Provider didn't return separate reasoning — content may
            # already contain <think> tags (some providers inline them)
            if "<think>" in content:
                return content
            # No reasoning found at all — return content as-is with warning
            return f"<!-- WARNING: no reasoning field returned -->\n{content}"

    elif raw:
        # Local vLLM: completions endpoint with chat template + <think> prefix
        prompt = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if not response.choices:
            raise RuntimeError(f"Empty response from API: {response}")
        return "<think>\n" + response.choices[0].text
    else:
        content = PROMPT_TEMPLATE.replace("{question}", question)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def process_question(
    client: OpenAI,
    model: str,
    q: dict,
    temperature: float,
    top_p: float,
    max_tokens: int,
    raw: bool,
    fout,
) -> dict:
    """Process a single question: call API, write result to file, return record."""
    start = time.perf_counter()
    try:
        raw_output = generate_trace(
            client, model, q["question"],
            temperature, top_p, max_tokens, raw=raw,
        )
        record = {
            **q,
            "model": model,
            "raw_output": raw_output,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timestamp": time.time(),
        }
    except Exception as e:
        record = {
            **q,
            "model": model,
            "raw_output": None,
            "error": str(e),
            "timestamp": time.time(),
        }

    elapsed = time.perf_counter() - start

    # Thread-safe write
    with _write_lock:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    status = "OK" if record.get("raw_output") else f"ERR: {record.get('error', '?')}"
    return {"id": q["id"], "status": status, "elapsed": round(elapsed, 1)}


def run_generation(
    client: OpenAI,
    model: str,
    questions: list[dict],
    output_path: Path,
    temperature: float,
    top_p: float,
    max_tokens: int,
    workers: int,
    resume: bool,
    raw: bool = False,
):
    """Generate traces for all questions using concurrent workers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-completed IDs
    done_ids = set()
    if resume and output_path.exists():
        # Only count successful traces (raw_output is not None) as done
        keep_lines = []
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("raw_output") is not None:
                    done_ids.add(rec["id"])
                    keep_lines.append(line)
        # Rewrite file without error records so they don't accumulate
        if len(keep_lines) < len(keep_lines) + 1:  # always rewrite to clean up
            with open(output_path, "w", encoding="utf-8") as f:
                f.writelines(keep_lines)
        print(f"Resuming: {len(done_ids)} successful traces kept")

    remaining = [q for q in questions if q["id"] not in done_ids]
    print(f"Generating {len(remaining)} traces ({len(done_ids)} already done)")
    print(f"Workers: {workers}\n")

    if not remaining:
        return

    success = 0
    errors = 0

    with open(output_path, "a", encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_question,
                    client, model, q,
                    temperature, top_p, max_tokens, raw,
                    fout,
                ): q
                for q in remaining
            }

            with tqdm(total=len(remaining), desc="Generating") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result["status"] == "OK":
                            success += 1
                        else:
                            errors += 1
                            print(f"  {result['id']}: {result['status']}")
                    except Exception as e:
                        q = futures[future]
                        errors += 1
                        print(f"  {q['id']}: unhandled exception: {e}")
                    pbar.update(1)

    print(f"\nCompleted: {success} OK, {errors} errors out of {len(remaining)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT traces via API (concurrent)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. deepseek-reasoner, Qwen/QwQ-32B)")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                        help="API base URL")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (falls back to OPENAI_API_KEY / OPENROUTER_API_KEY env vars)")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "arc_challenge"],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--n-per-dataset", type=int, default=200)
    parser.add_argument("--output", type=str, default="runs/traces.jsonl")
    parser.add_argument("--temperature", type=float, default=SAMPLING_DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=SAMPLING_DEFAULTS["top_p"])
    parser.add_argument("--max-tokens", type=int, default=SAMPLING_DEFAULTS["max_tokens"])
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent API workers (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw", action="store_true",
                        help="Send bare question (let model use native <think> tags)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions already in the output file")
    args = parser.parse_args()

    # Resolve API key: CLI flag > env vars
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = "EMPTY"  # local vLLM default

    client = OpenAI(base_url=args.api_base, api_key=api_key)

    # Load questions
    all_questions = []
    for ds_name in args.datasets:
        print(f"Loading {args.n_per_dataset} questions from {ds_name}...")
        questions = load_questions(ds_name, n=args.n_per_dataset, seed=args.seed)
        all_questions.extend(questions)
        print(f"  Loaded {len(questions)} questions")

    print(f"\nTotal: {len(all_questions)} questions across {len(args.datasets)} datasets")
    print(f"Model: {args.model}")
    print(f"Server: {args.api_base}")
    print(f"Output: {args.output}")

    run_generation(
        client=client,
        model=args.model,
        questions=all_questions,
        output_path=Path(args.output),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        workers=args.workers,
        resume=args.resume,
        raw=args.raw,
    )

    print(f"Done. Traces written to {args.output}")


if __name__ == "__main__":
    main()