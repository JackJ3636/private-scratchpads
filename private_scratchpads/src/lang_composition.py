"""Compute B5 language-composition stats for the ARR checklist / Appendix A.

Reads the merged analysis set (filter.py's output: 83 illegible traces) and
reports, for the paragraph "Language composition of traces" in
arr_submission.tex:
  - how many traces contain at least one non-Latin script character
  - how many contain Han (Chinese) characters
  - the median per-trace percentage of characters outside the ASCII range

"Non-Latin script" deliberately excludes the math/science Unicode ranges
that scrub.py already treats as legitimate domain notation (Greek letters,
math operators, arrows, etc.) rather than language-mixing, so this stays
consistent with how the rest of the paper draws that line.

Usage:
    python src/lang_composition.py --analysis-set runs/analysis_set.jsonl
"""

import argparse
import json
import unicodedata

# Same ranges as scrub.py's _MATH_SCIENCE_RANGES — kept in sync deliberately.
_MATH_SCIENCE_RANGES = (
    (0x00B0, 0x00BF), (0x00C0, 0x00FF), (0x00D7, 0x00D7), (0x00F7, 0x00F7),
    (0x0370, 0x03FF), (0x2070, 0x209F), (0x2100, 0x214F), (0x2150, 0x218F),
    (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2300, 0x23FF), (0x27C0, 0x27EF),
    (0x2980, 0x29FF), (0x2A00, 0x2AFF),
)

_HAN_RANGES = (
    (0x2E80, 0x2EFF),   # CJK Radicals Supplement
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),   # Halfwidth/Fullwidth forms
)


def _in_ranges(cp, ranges):
    return any(lo <= cp <= hi for lo, hi in ranges)


def _is_math_science(c):
    return _in_ranges(ord(c), _MATH_SCIENCE_RANGES)


def _is_han(c):
    return _in_ranges(ord(c), _HAN_RANGES)


def has_non_latin_script(text):
    """True if text has a non-ASCII letter that isn't math/science notation."""
    for c in text:
        cp = ord(c)
        if cp < 128 or _is_math_science(c):
            continue
        if unicodedata.category(c).startswith("L"):
            return True
    return False


def has_han(text):
    return any(_is_han(c) for c in text)


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def main():
    parser = argparse.ArgumentParser(description="Compute B5 language-composition stats")
    parser.add_argument("--analysis-set", type=str, required=True,
                         help="Merged analysis set JSONL (filter.py output)")
    args = parser.parse_args()

    traces = []
    with open(args.analysis_set, encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))

    n = len(traces)
    non_latin_count = 0
    han_count = 0
    non_ascii_pcts = []

    for t in traces:
        cot = t.get("cot_text", "") or ""
        if has_non_latin_script(cot):
            non_latin_count += 1
        if has_han(cot):
            han_count += 1
        # Prefer the heuristic already computed by parse_traces.py if present.
        frac = t.get("heuristics", {}).get("non_ascii_frac")
        if frac is None and cot:
            frac = sum(1 for c in cot if ord(c) > 127) / len(cot)
        non_ascii_pcts.append(100 * (frac or 0.0))

    print(f"Traces analysed: {n}")
    print(f"Contain >=1 non-Latin script character (excl. math/science notation): {non_latin_count}")
    print(f"Contain Han (Chinese) characters: {han_count}")
    print(f"Median % of characters outside ASCII range: {median(non_ascii_pcts):.1f}%")
    print()
    print("Paste into arr_submission.tex Appendix A as:")
    print(f'  of the {n} illegible traces, {non_latin_count} contain at least one non-Latin '
          f'script\n  character and {han_count} contain Han characters, with a median of '
          f'{median(non_ascii_pcts):.1f}\\% of\n  characters per trace outside the ASCII range.')


if __name__ == "__main__":
    main()
