"""Phase 1: Generate CoT traces from reasoning models.

Loads questions from GSM8K and ARC-Challenge, prompts a vLLM server
via its OpenAI-compatible API, and saves raw traces to runs/traces.jsonl.

Usage:
    # Start vLLM on your cloud GPU:
    #   vllm serve Qwen/QwQ-32B --tensor-parallel-size 2

    # With prompt template (wraps question in scratchpad instructions):
    python src/generate.py \
        --model Qwen/QwQ-32B \
        --api-base http://<gpu-host>:8000/v1 \
        --datasets gsm8k arc_challenge \
        --n-per-dataset 200 \
        --output runs/traces.jsonl

    # Raw mode (sends bare question, lets model use native <think> tags):
    python src/generate.py \
        --model Qwen/QwQ-32B \
        --api-base http://<gpu-host>:8000/v1 \
        --raw \
        --output runs/traces.jsonl
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

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
        "answer_field": "Correct Answer",  # gold is the correct answer text
        "parse_gold": lambda a: a.strip(),
    },
}


def load_questions(dataset: str, n: int, seed: int = 42) -> list[dict]:
    """Load and sample n questions from a HuggingFace dataset."""
    cfg = DATASET_CONFIGS[dataset]
    ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])

    # Sample n items (or take all if dataset is smaller)
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


def generate_trace(
    client: OpenAI,
    model: str,
    question: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    raw: bool = False,
) -> str:
    """Send a single question to the vLLM server and return the raw output.

    If raw=True, uses the /v1/completions endpoint with the chat template
    manually applied and a <think> prefix to trigger the model's native
    reasoning mode (e.g. QwQ / DeepSeek-R1).
    Otherwise wraps in the scratchpad prompt template via chat completions.
    """
    if raw and "deepseek" in model.lower():
        # DeepSeek-R1 API / OpenRouter: reasoning comes in reasoning_content
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
        reasoning = getattr(msg, "reasoning_content", None)
        # OpenRouter puts reasoning in .reasoning instead of .reasoning_content
        if reasoning is None:
            reasoning = getattr(msg, "reasoning", None) or ""
        content = msg.content or ""
        return f"<think>\n{reasoning}\n</think>\n{content}"
    elif raw:
        # Local vLLM: use completions endpoint with chat template + <think> prefix
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


def run_generation(
    client: OpenAI,
    model: str,
    questions: list[dict],
    output_path: Path,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    resume: bool,
    raw: bool = False,
):
    """Generate traces for all questions, appending to output file.

    Supports resume: skips questions whose IDs already appear in the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed IDs for resume
    done_ids = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                done_ids.add(record["id"])
        print(f"Resuming: {len(done_ids)} traces already completed")

    remaining = [q for q in questions if q["id"] not in done_ids]
    print(f"Generating {len(remaining)} traces ({len(done_ids)} already done)")

    # Process in batches for progress tracking
    with open(output_path, "a", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(remaining), batch_size), desc="Batches"):
            batch = remaining[i : i + batch_size]

            for q in batch:
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
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()

                except Exception as e:
                    print(f"Error on {q['id']}: {e}")
                    # Write a failed record so we can retry later
                    record = {
                        **q,
                        "model": model,
                        "raw_output": None,
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()


def main():
    parser = argparse.ArgumentParser(description="Generate CoT traces via vLLM")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name as served by vLLM (e.g. Qwen/QwQ-32B)")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                        help="vLLM server base URL")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (falls back to OPENAI_API_KEY / OPENROUTER_API_KEY env vars)")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "arc_challenge"],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--n-per-dataset", type=int, default=200)
    parser.add_argument("--output", type=str, default="runs/traces.jsonl")
    parser.add_argument("--temperature", type=float, default=SAMPLING_DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=SAMPLING_DEFAULTS["top_p"])
    parser.add_argument("--max-tokens", type=int, default=SAMPLING_DEFAULTS["max_tokens"])
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of questions per progress batch")
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

    # Load questions from all requested datasets
    all_questions = []
    for ds_name in args.datasets:
        print(f"Loading {args.n_per_dataset} questions from {ds_name}...")
        questions = load_questions(ds_name, n=args.n_per_dataset, seed=args.seed)
        all_questions.extend(questions)
        print(f"  Loaded {len(questions)} questions")

    print(f"\nTotal: {len(all_questions)} questions across {len(args.datasets)} datasets")
    print(f"Model: {args.model}")
    print(f"Server: {args.api_base}")
    print(f"Output: {args.output}\n")

    run_generation(
        client=client,
        model=args.model,
        questions=all_questions,
        output_path=Path(args.output),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        resume=args.resume,
        raw=args.raw,
    )

    print(f"\nDone. Traces written to {args.output}")


if __name__ == "__main__":
    main()
