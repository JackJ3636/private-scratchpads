# Private Scratchpads

Code for the *Private Scratchpads* experiments: when a reasoning model produces an
**illegible chain-of-thought** (CoT) yet still answers correctly, what kind of channel
is the illegible content carrying?

The pipeline generates CoT traces from a reasoning model (QwQ-32B), filters for
illegible-but-correct traces on GPQA Diamond, then has multiple readers (self,
same-family, cross-family) try to recover the final answer from the CoT alone.
It also runs character-class scrubbing experiments (S0–S7) and a translation
study.

## Headline finding

Cross-family readers (GPT-4o) decode illegible CoT **better** than the model that
generated it (self-reader). The illegible content is not a private language — it
behaves like compressed-but-generic text, with non-ASCII characters that actively
hinder the self-reader.

## Repository layout

```
private_scratchpads/
├── prompts/               # generator, reader, translator prompts
├── scripts/               # shell helpers (e.g. run_generate.sh)
└── src/
    ├── generate.py             # Phase 1: generate CoT traces via OpenAI-compatible API
    ├── generate_concurrent.py  # concurrent variant
    ├── parse_traces.py         # extract <think> content + final answer
    ├── grade_correctness.py    # GPT-4o correctness grading
    ├── assess.py               # GPT-4o illegibility assessment (1–10)
    ├── filter.py               # build the analysis set (illegible + correct)
    ├── decode.py               # Phase 2: reader decoding
    ├── scrub.py                # S0–S7 scrub transforms
    ├── scrub_decode.py         # Phase 3: scrub + decode pipeline
    ├── score.py                # grading, results tables, McNemar tests
    ├── translate.py            # translation experiment (3 stages)
    ├── figures.py              # generate the 5 paper figures
    ├── autograder_gpt.py       # GPT-4o autograder utility
    └── autograder_utils.py     # illegibility example prompts
```

GPQA Diamond is loaded from HuggingFace at runtime — there's no local `data/`
directory to populate.

Outputs land in `private_scratchpads/runs/` (intermediate JSONL) and
`private_scratchpads/results/` (figures + tables). Both are gitignored —
regenerate them by running the pipeline below.

## Setup

Python 3.10+ is recommended.

```bash
pip install -r private_scratchpads/requirements.txt
```

Set the API keys you need as environment variables:

```bash
export OPENAI_API_KEY=...        # GPT-4o (judge, autograder, cross-reader)
export OPENROUTER_API_KEY=...    # QwQ-32B, Qwen3-32B, Claude 3.5 Haiku
```

Only the keys for the providers you actually call are required.

## Pipeline

All commands are run from the `private_scratchpads/` directory.

### 1. Generate CoT traces

```bash
python src/generate.py \
    --model qwen/qwq-32b \
    --provider openrouter \
    --dataset gpqa \
    --output runs/qwq_openrouter_v2.jsonl
```

`generate_concurrent.py` is a faster, parallel variant of the same script.

### 2. Parse, grade, assess, filter

```bash
python src/parse_traces.py       --input runs/qwq_openrouter_v2.jsonl --output runs/parsed_qwq.jsonl
python src/grade_correctness.py  --input runs/parsed_qwq.jsonl        --output runs/graded_qwq.jsonl
python src/assess.py             --input runs/parsed_qwq.jsonl        --output runs/assessed_qwq.jsonl
python src/filter.py             --min-illegibility 5 --correctness correct partially_correct \
                                 --output runs/analysis_set.jsonl
```

### 3. Reader decoding (Phase 2)

```bash
python src/decode.py --condition c1 --reader qwq         --output runs/decoded.jsonl
python src/decode.py --condition c3 --reader gpt-4o      --output runs/decoded.jsonl
# ...etc for c2, c4, c5, c6, c7
python src/score.py  --input runs/decoded.jsonl          --output runs/scored.jsonl
```

### 4. Scrubbing (Phase 3)

```bash
python src/scrub_decode.py --scrubs s0 s1 s2 s3 s4 s5 s6 s7 \
                           --output runs/scrub_decoded.jsonl
python src/score.py        --input  runs/scrub_decoded.jsonl \
                           --output runs/scrub_scored.jsonl
```

### 5. Translation experiment

```bash
python src/translate.py --output runs/translated.jsonl
python src/translate.py --judge  --input  runs/translated.jsonl \
                                 --output runs/translated_judged.jsonl
```

### 6. Figures

```bash
python src/figures.py --output results/figures/
```

Exact flags for each script are visible via `--help`; the snippets above show the
shape of the pipeline rather than every option.

## Models used

| Role | Model | Provider |
|---|---|---|
| Generator | QwQ-32B (`qwen/qwq-32b`) | OpenRouter |
| Self-reader | QwQ-32B | OpenRouter |
| Same-family cross-reader | Qwen3-32B (`qwen/qwen3-32b`) | OpenRouter |
| Cross-reader (strong) | GPT-4o | OpenAI |
| Cross-reader (weak) | Claude 3.5 Haiku | OpenRouter |
| Judge / autograder | GPT-4o | OpenAI |

## Dataset

[GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) — 198 graduate-level
science questions, presented free-text (without the multiple-choice answer key) so
all scoring goes through GPT-4o.

## Released traces

[`private_scratchpads/release/`](private_scratchpads/release/) contains the 83
illegible and 11 legible reasoning traces behind the paper's main experiment, with
GPQA question text and gold answers removed — see that directory's README for
what's included and why some question content may still be paraphrased inline in
the reasoning text.

## Notes

- `runs/` and `results/` are gitignored — clone the repo and re-run the pipeline
  to regenerate them.
