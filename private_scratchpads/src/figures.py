"""Generate plots and tables for the paper.

Figures:
  1. Grouped bar chart: self vs cross accuracy by condition
  2. Scrubbing sensitivity curves
  3. Illegibility score vs self-decode accuracy (bar chart with CI)
  4. (Optional) Self-cross accuracy gap histogram
  5. Translation quality comparison

Usage:
    python src/figures.py \
        --scored runs/scored.jsonl \
        --scored-extra runs/scored_haiku.jsonl runs/scored_qwen_cross.jsonl runs/scored_c7.jsonl \
        --scrub-scored runs/scrub_scored.jsonl \
        --analysis-set runs/analysis_set.jsonl \
        --output-dir results/figures
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------- global style ----------

# Okabe-Ito colorblind-safe palette
OI_BLUE       = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_GREEN      = "#009E73"
OI_SKY        = "#56B4E9"
OI_ORANGE     = "#E69F00"
OI_PURPLE     = "#CC79A7"
OI_GRAY       = "#999999"


def setup_paper_style():
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        # Font
        "font.family":          "sans-serif",
        "font.size":            10,
        "axes.labelsize":       10,
        "axes.titlesize":       11,
        "axes.titleweight":     "normal",
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "legend.title_fontsize": 9,
        # Axes
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.8,
        "axes.edgecolor":       "#444444",
        "axes.labelcolor":      "#222222",
        "xtick.color":          "#444444",
        "ytick.color":          "#444444",
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        # Grid
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "grid.color":           "#DDDDDD",
        "grid.linewidth":       0.6,
        # Figure
        "figure.dpi":           300,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
    })


# ---------- helpers ----------

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bootstrap_ci(correct, n_resamples=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.array(correct, dtype=float)
    acc = arr.mean()
    boot = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    return float(acc), float(np.percentile(boot, 100 * alpha)), float(np.percentile(boot, 100 * (1 - alpha)))


def group_accuracy(records, group_key):
    """Group records by group_key function, return {key: (acc, ci_lo, ci_hi, n)}."""
    groups = defaultdict(list)
    for r in records:
        is_correct = r.get("correctness") == "correct"
        groups[group_key(r)].append(is_correct)
    results = {}
    for key, vals in groups.items():
        acc, lo, hi = bootstrap_ci(vals)
        results[key] = (acc, lo, hi, len(vals))
    return results


# ---------- Figure 1: Decoding accuracy by condition ----------

def figure1_decoding_accuracy(records, output_path):
    """Hero figure: grouped bar chart of self/cross accuracy by condition."""

    stats = group_accuracy(records, lambda r: (r["condition"], r["reader"], r.get("reader_model", "")))

    condition_order = ["C1", "C4", "C3", "C5", "C2", "C7", "C6"]
    condition_labels = {
        "C1": "Self\n(no Q)",
        "C4": "Self\n(+Q)",
        "C3": "Cross\n(no Q)",
        "C5": "Cross\n(+Q)",
        "C2": "Same-family\n(no Q)",
        "C7": "Legible\n(self)",
        "C6": "Random\nCoT",
    }

    colours = {
        ("self",  "qwen/qwq-32b"):                    OI_BLUE,
        ("cross", "gpt-4o"):                           OI_VERMILLION,
        ("cross", "anthropic/claude-3.5-haiku"):       OI_GREEN,
        ("cross", "qwen/qwen3-32b"):                   OI_ORANGE,
    }
    default_colour = OI_GRAY

    cond_data = defaultdict(list)
    for (cond, reader, model), (acc, lo, hi, n) in stats.items():
        cond_data[cond].append((reader, model, acc, lo, hi, n))

    x_positions = []
    x_labels = []
    bar_data = []
    pos = 0
    bar_width = 0.55

    for cond in condition_order:
        if cond not in cond_data:
            continue
        entries = sorted(cond_data[cond], key=lambda e: (e[0], e[1]))
        n_bars = len(entries)
        offsets = (
            np.linspace(-bar_width * (n_bars - 1) / 2,
                         bar_width * (n_bars - 1) / 2, n_bars)
            if n_bars > 1 else [0]
        )
        for i, (reader, model, acc, lo, hi, n) in enumerate(entries):
            colour = colours.get((reader, model), default_colour)
            x = pos + offsets[i]
            bar_data.append((x, acc, acc - lo, hi - acc, colour, reader, model, n))

        x_positions.append(pos)
        x_labels.append(condition_labels.get(cond, cond))
        pos += max(n_bars, 1) * bar_width + 0.5

    fig, ax = plt.subplots(figsize=(11, 4.5))

    legend_handles = {}
    for x, acc, err_lo, err_hi, colour, reader, model, n in bar_data:
        short_model = model.split("/")[-1] if "/" in model else model
        label = f"{reader} ({short_model})"
        bar = ax.bar(x, acc, width=bar_width * 0.82, color=colour,
                     edgecolor="white", linewidth=0.4, alpha=0.88, zorder=3)
        ax.errorbar(x, acc, yerr=[[err_lo], [err_hi]], fmt="none",
                    ecolor="#333333", capsize=2.5, linewidth=0.9, zorder=4)
        ax.text(x, acc + err_hi + 0.015, f"{acc:.0%}",
                ha="center", va="bottom", fontsize=7.5, color="#333333")
        if label not in legend_handles:
            legend_handles[label] = bar

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Reader Decoding Accuracy by Condition")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)

    ax.legend(
        legend_handles.values(), legend_handles.keys(),
        loc="upper right", frameon=True, framealpha=0.9,
        edgecolor="#CCCCCC", borderpad=0.6,
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved Figure 1 -> {output_path}")


# ---------- Figure 2: Scrubbing sensitivity curves ----------

def figure2_scrubbing_curves(records, output_path):
    """Scrubbing sensitivity: accuracy vs scrub level for self and cross readers."""

    stats = group_accuracy(records, lambda r: (r["scrub"], r["reader"]))

    scrub_order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    scrub_labels = {
        "S0": "None",
        "S1": "NFKC",
        "S2": "Rm Ctrl",
        "S3": "ASCII\nOnly",
        "S4": "ASCII\nReplace",
        "S5": "Random\nNon-ASCII",
        "S6": "Collapse\nWS",
        "S7": "Entropy\nDropout",
    }

    readers = sorted(set(r for (_, r) in stats.keys()))
    colours = {"self": OI_BLUE, "cross": OI_VERMILLION}

    fig, ax = plt.subplots(figsize=(9, 4))

    for reader in readers:
        xs, accs, los, his = [], [], [], []
        for i, scrub in enumerate(scrub_order):
            key = (scrub, reader)
            if key in stats:
                acc, lo, hi, n = stats[key]
                xs.append(i)
                accs.append(acc)
                los.append(lo)
                his.append(hi)

        if not xs:
            continue

        accs, los, his = np.array(accs), np.array(los), np.array(his)
        colour = colours.get(reader, OI_GRAY)
        ax.plot(xs, accs, "o-", color=colour, label=reader,
                linewidth=1.8, markersize=5, zorder=3)
        ax.fill_between(xs, los, his, color=colour, alpha=0.12, zorder=2)

    ax.set_xticks(range(len(scrub_order)))
    ax.set_xticklabels([scrub_labels.get(s, s) for s in scrub_order])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Scrub Transform")
    ax.set_title("Scrubbing Sensitivity: Self vs. Cross Reader")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved Figure 2 -> {output_path}")


# ---------- Figure 3: Illegibility score vs self-decode accuracy ----------

def figure3_illegibility_bars(analysis_records, scored_records, output_path):
    """Bar chart: mean self-decode accuracy by illegibility score bin, with 95% CI."""

    # Build per-trace self-reader correctness from C1 (self, no-q)
    self_correct = {}
    for r in scored_records:
        if r["condition"] == "C1" and r["reader"] == "self":
            self_correct[r["trace_id"]] = 1.0 if r.get("correctness") == "correct" else 0.0

    # Get illegibility score from analysis set
    illeg_scores = {}
    for r in analysis_records:
        illeg_scores[r["id"]] = r.get("illegibility_score", None)

    # Bin by illegibility score
    score_to_correct = defaultdict(list)
    for tid, score in illeg_scores.items():
        if score is not None and tid in self_correct:
            score_to_correct[int(score)].append(self_correct[tid])

    if not score_to_correct:
        print("  Skipping Figure 3: no matching data")
        return

    scores_sorted = sorted(score_to_correct.keys())
    accs, lows, highs, counts = [], [], [], []
    for s in scores_sorted:
        vals = score_to_correct[s]
        if len(vals) >= 2:
            acc, lo, hi = bootstrap_ci(vals)
        else:
            acc, lo, hi = np.mean(vals), np.mean(vals), np.mean(vals)
        accs.append(acc)
        lows.append(lo)
        highs.append(hi)
        counts.append(len(vals))

    xs = np.arange(len(scores_sorted))
    err_lo = np.array(accs) - np.array(lows)
    err_hi = np.array(highs) - np.array(accs)

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(xs, accs, color=OI_BLUE, alpha=0.82,
                  edgecolor="white", linewidth=0.4, zorder=3)
    ax.errorbar(xs, accs, yerr=[err_lo, err_hi], fmt="none",
                ecolor="#333333", capsize=3, linewidth=0.9, zorder=4)

    # Annotate count below each bar label
    for i, (s, c) in enumerate(zip(scores_sorted, counts)):
        ax.text(i, -0.06, f"n={c}", ha="center", va="top",
                fontsize=7.5, color="#666666", transform=ax.get_xaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in scores_sorted])
    ax.set_xlabel("Illegibility Score (1 = clear, 10 = opaque)")
    ax.set_ylabel("Self-Reader Accuracy (C1)")
    ax.set_title("Accuracy vs. Illegibility Score")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved Figure 3 -> {output_path}")


# ---------- Figure 4: Per-trace self-cross gap histogram ----------

def figure4_gap_histogram(scored_records, output_path):
    """Bar chart of per-trace (self_correct - cross_correct) categories."""

    self_by_id = {}
    cross_by_id = {}
    for r in scored_records:
        is_correct = 1 if r.get("correctness") == "correct" else 0
        if r["condition"] == "C1" and r["reader"] == "self":
            self_by_id[r["trace_id"]] = is_correct
        elif r["condition"] == "C3" and r["reader"] == "cross":
            cross_by_id[r["trace_id"]] = is_correct

    shared = sorted(set(self_by_id) & set(cross_by_id))
    if len(shared) < 5:
        print(f"  Skipping Figure 4: only {len(shared)} paired traces")
        return

    gaps = [self_by_id[tid] - cross_by_id[tid] for tid in shared]
    counts = {-1: gaps.count(-1), 0: gaps.count(0), 1: gaps.count(1)}

    fig, ax = plt.subplots(figsize=(5, 3.8))
    labels = ["Cross only\ncorrect", "Both\nsame", "Self only\ncorrect"]
    values = [counts[-1], counts[0], counts[1]]
    colours = [OI_VERMILLION, OI_GRAY, OI_BLUE]

    ax.bar(labels, values, color=colours, edgecolor="white",
           linewidth=0.4, width=0.55, alpha=0.88, zorder=3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.4, str(v), ha="center", va="bottom",
                fontsize=9.5, color="#222222")

    ax.set_ylabel("Number of Traces")
    ax.set_title("Per-Trace Agreement: Self (C1) vs. Cross (C3)")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved Figure 4 -> {output_path}")


# ---------- Figure 5: Translation quality comparison ----------

def figure5_translation_quality(records, output_path):
    """Grouped bar chart: coherence & faithfulness for self vs cross translator."""

    by_reader = {}
    for r in records:
        if r.get("coherence") is None:
            continue
        model = r["reader_model"].split("/")[-1]
        label = f"{r['reader']}\n({model})"
        by_reader.setdefault(label, []).append(r)

    if not by_reader:
        print("  Skipping Figure 5: no valid judged translations")
        return

    readers = sorted(by_reader.keys())
    coherence_means, faith_means = [], []
    coherence_cis, faith_cis = [], []

    for reader in readers:
        recs = by_reader[reader]
        coh = [r["coherence"] for r in recs]
        faith = [r["faithfulness"] for r in recs]
        coh_mean, coh_lo, coh_hi = bootstrap_ci(coh)
        faith_mean, faith_lo, faith_hi = bootstrap_ci(faith)
        coherence_means.append(coh_mean)
        faith_means.append(faith_mean)
        coherence_cis.append((coh_mean - coh_lo, coh_hi - coh_mean))
        faith_cis.append((faith_mean - faith_lo, faith_hi - faith_mean))

    x = np.arange(len(readers))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.bar(x - width / 2, coherence_means, width, color=OI_BLUE,
           label="Coherence", edgecolor="white", linewidth=0.4, alpha=0.88, zorder=3)
    ax.errorbar(x - width / 2, coherence_means,
                yerr=list(zip(*coherence_cis)),
                fmt="none", ecolor="#333333", capsize=3, linewidth=0.9, zorder=4)

    ax.bar(x + width / 2, faith_means, width, color=OI_VERMILLION,
           label="Faithfulness", edgecolor="white", linewidth=0.4, alpha=0.88, zorder=3)
    ax.errorbar(x + width / 2, faith_means,
                yerr=list(zip(*faith_cis)),
                fmt="none", ecolor="#333333", capsize=3, linewidth=0.9, zorder=4)

    for i, (c, f) in enumerate(zip(coherence_means, faith_means)):
        ax.text(i - width / 2, c + coherence_cis[i][1] + 0.06, f"{c:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="#333333")
        ax.text(i + width / 2, f + faith_cis[i][1] + 0.06, f"{f:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(readers)
    ax.set_ylabel("Score (1–5)")
    ax.set_title("Translation Quality: Self vs. Cross Reader")
    ax.set_ylim(0, 5.8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", loc="lower right")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved Figure 5 -> {output_path}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--scored", type=str, default="runs/scored.jsonl",
                        help="Primary scored JSONL (C1-C6)")
    parser.add_argument("--scored-extra", nargs="*", default=[],
                        help="Additional scored JSONLs (haiku, qwen, c7)")
    parser.add_argument("--scrub-scored", type=str, default=None,
                        help="Scored scrub results JSONL")
    parser.add_argument("--analysis-set", type=str, default="runs/analysis_set.jsonl",
                        help="Analysis set JSONL (for illegibility scores)")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--translation-judged", type=str, default=None,
                        help="Judged translation JSONL")
    parser.add_argument("--figures", nargs="*", default=["1", "2", "3", "4", "5"],
                        help="Which figures to generate (1-5)")
    args = parser.parse_args()

    setup_paper_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all scored records
    all_scored = []
    if Path(args.scored).exists():
        all_scored.extend(load_jsonl(args.scored))
        print(f"Loaded {len(all_scored)} records from {args.scored}")
    for extra in args.scored_extra:
        if Path(extra).exists():
            recs = load_jsonl(extra)
            all_scored.extend(recs)
            print(f"Loaded {len(recs)} records from {extra}")
    print(f"Total scored records: {len(all_scored)}")

    # Figure 1
    if "1" in args.figures and all_scored:
        print("\nGenerating Figure 1: Decoding Accuracy...")
        figure1_decoding_accuracy(all_scored, output_dir / "fig1_decoding_accuracy.png")

    # Figure 2
    if "2" in args.figures:
        if args.scrub_scored and Path(args.scrub_scored).exists():
            scrub_records = load_jsonl(args.scrub_scored)
            print(f"\nGenerating Figure 2: Scrubbing Curves ({len(scrub_records)} records)...")
            figure2_scrubbing_curves(scrub_records, output_dir / "fig2_scrubbing_curves.png")
        else:
            print("\nSkipping Figure 2: no --scrub-scored provided or file missing")

    # Figure 3
    if "3" in args.figures:
        analysis_path = Path(args.analysis_set)
        if analysis_path.exists() and all_scored:
            analysis_records = load_jsonl(args.analysis_set)
            print(f"\nGenerating Figure 3: Illegibility Bars ({len(analysis_records)} traces)...")
            figure3_illegibility_bars(analysis_records, all_scored,
                                      output_dir / "fig3_illegibility_scatter.png")
        else:
            print("\nSkipping Figure 3: missing analysis set or scored data")

    # Figure 4
    if "4" in args.figures and all_scored:
        print("\nGenerating Figure 4: Self-Cross Gap Histogram...")
        figure4_gap_histogram(all_scored, output_dir / "fig4_gap_histogram.png")

    # Figure 5
    if "5" in args.figures:
        if args.translation_judged and Path(args.translation_judged).exists():
            trans_records = load_jsonl(args.translation_judged)
            print(f"\nGenerating Figure 5: Translation Quality ({len(trans_records)} records)...")
            figure5_translation_quality(trans_records, output_dir / "fig5_translation_quality.png")
        else:
            print("\nSkipping Figure 5: no --translation-judged provided or file missing")

    print("\nDone.")


if __name__ == "__main__":
    main()
