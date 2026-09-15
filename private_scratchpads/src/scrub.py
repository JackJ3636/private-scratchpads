"""Phase 3: Apply controlled scrubbing transforms to CoT text.

Scrub definitions:
  S0: None (baseline)
  S1: Unicode NFKC normalization
  S2: Remove all Unicode Cf category chars
  S3: ASCII-only (delete non-ASCII)
  S4: ASCII-replace (non-ASCII -> '?')
  S5: Random non-ASCII replacement
  S6: Collapse whitespace
  S7: Entropy-based span dropout (top 10%)
"""

import math
import random
import re
import unicodedata
from collections import Counter

# Characters commonly used in scientific/mathematical reasoning that should
# survive ASCII-based scrubs. These are legitimate notation, not illegible encoding.
_MATH_SCIENCE_RANGES = (
    (0x00B0, 0x00BF),  # ° ± ² ³ µ · ¹ » ¼ ½ ¾
    (0x00C0, 0x00FF),  # Latin accented (Ångström, naïve, résumé, etc.)
    (0x00D7, 0x00D7),  # ×
    (0x00F7, 0x00F7),  # ÷
    (0x0370, 0x03FF),  # Greek and Coptic (α β γ δ π Σ Ω etc.)
    (0x2070, 0x209F),  # Superscripts and subscripts
    (0x2100, 0x214F),  # Letterlike symbols (ℏ ℝ ℤ ℂ etc.)
    (0x2150, 0x218F),  # Number forms (fractions)
    (0x2190, 0x21FF),  # Arrows (→ ← ↔ ⇒ etc.)
    (0x2200, 0x22FF),  # Mathematical operators (∈ ∀ ∃ ∑ ∏ √ ∞ ≤ ≥ ≠ ≈ etc.)
    (0x2300, 0x23FF),  # Misc technical (⟨ ⟩ etc.)
    (0x27C0, 0x27EF),  # Misc mathematical symbols-A
    (0x2980, 0x29FF),  # Misc mathematical symbols-B
    (0x2A00, 0x2AFF),  # Supplemental mathematical operators
)


def _is_math_science(c: str) -> bool:
    """Return True if the character is in a mathematical/scientific Unicode range."""
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _MATH_SCIENCE_RANGES)


def scrub_s0_none(text: str) -> str:
    """S0: No scrubbing (baseline)."""
    return text


def scrub_s1_nfkc(text: str) -> str:
    """S1: Unicode NFKC normalization."""
    return unicodedata.normalize("NFKC", text)


def scrub_s2_remove_format_chars(text: str) -> str:
    """S2: Remove all Unicode Cf (format) and Cc (control) category characters."""
    return "".join(c for c in text if unicodedata.category(c) not in ("Cf", "Cc"))


def scrub_s3_ascii_only(text: str) -> str:
    """S3: Delete non-ASCII characters, preserving Greek/math/science symbols."""
    return "".join(c for c in text if ord(c) < 128 or _is_math_science(c))


def scrub_s4_ascii_replace(text: str) -> str:
    """S4: Replace non-ASCII characters with '?', preserving Greek/math/science symbols."""
    return "".join(c if ord(c) < 128 or _is_math_science(c) else "?" for c in text)


def scrub_s5_random_non_ascii(text: str, seed: int = 42) -> str:
    """S5: Replace each non-ASCII char with a random non-ASCII char.

    Controls for tokenizer artifacts vs. genuine encoding.
    """
    rng = random.Random(seed)
    # Pool of random non-ASCII codepoints (common CJK + Cyrillic + Arabic range)
    pool = [chr(cp) for cp in range(0x4E00, 0x4E00 + 500)]

    result = []
    for c in text:
        if ord(c) >= 128:
            result.append(rng.choice(pool))
        else:
            result.append(c)
    return "".join(result)


def scrub_s6_collapse_whitespace(text: str) -> str:
    """S6: Normalize all whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def scrub_s7_entropy_dropout(text: str, dropout_frac: float = 0.10) -> str:
    """S7: Replace the top dropout_frac highest-entropy character spans with [REDACTED].

    Uses character unigram surprise as a proxy for entropy.
    """
    if not text:
        return text

    # Compute unigram frequencies
    counts = Counter(text)
    total = len(text)

    # Per-character surprise (-log2 p)
    surprises = []
    for i, c in enumerate(text):
        p = counts[c] / total
        surprise = -math.log2(p) if p > 0 else 0
        surprises.append((i, surprise))

    # Find threshold for top dropout_frac
    sorted_surprises = sorted(s for _, s in surprises)
    cutoff_idx = int(len(sorted_surprises) * (1 - dropout_frac))
    threshold = sorted_surprises[cutoff_idx] if cutoff_idx < len(sorted_surprises) else float("inf")

    # Replace high-surprise characters
    result = []
    in_redacted = False
    for i, c in enumerate(text):
        if surprises[i][1] >= threshold:
            if not in_redacted:
                result.append("[REDACTED]")
                in_redacted = True
        else:
            result.append(c)
            in_redacted = False

    return "".join(result)


SCRUBS = {
    "S0": scrub_s0_none,
    "S1": scrub_s1_nfkc,
    "S2": scrub_s2_remove_format_chars,
    "S3": scrub_s3_ascii_only,
    "S4": scrub_s4_ascii_replace,
    "S5": scrub_s5_random_non_ascii,
    "S6": scrub_s6_collapse_whitespace,
    "S7": scrub_s7_entropy_dropout,
}


def apply_scrub(text: str, scrub_id: str, **kwargs) -> str:
    """Apply a named scrub to CoT text."""
    if scrub_id not in SCRUBS:
        raise ValueError(f"Unknown scrub: {scrub_id}. Valid: {list(SCRUBS.keys())}")
    return SCRUBS[scrub_id](text, **kwargs) if scrub_id in ("S5", "S7") else SCRUBS[scrub_id](text)
