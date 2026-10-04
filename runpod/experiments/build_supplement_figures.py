"""Build supplementary measurement-reliability figures from the complete formal run.

Requires Python, numpy, pandas, matplotlib, PyMuPDF and the nature-figure QA
scripts. Run from any directory; --project and --skill-dir override defaults.
Original run files are never written. All stochastic repeats are displayed
individually; no uncertainty band is inferred for these descriptive comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from figure_paths import export_path, validation_path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


TEAL = "#5EA8A1"
DARK_TEAL = "#267F7B"
GRAY = "#777777"
LIGHT_GRAY = "#D9DEDD"
WIDTH_MM = 183
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.unicode_minus": True,
})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def describe_pair(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return {
        "n": len(x),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "spearman_rho": float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]),
        "mean_y_minus_x": float(np.mean(y - x)),
        "mean_absolute_difference": float(np.mean(np.abs(y - x))),
        "root_mean_square_difference": float(np.sqrt(np.mean((y - x) ** 2))),
        "strict_sign_reversal_count": int(np.count_nonzero(x * y < 0)),
        "strict_sign_reversal_fraction": float(np.mean(x * y < 0)),
        "zero_in_either_order_or_seed_count": int(np.count_nonzero((x == 0) | (y == 0))),
    }


def require_finite(df, columns):
    require(np.isfinite(df[columns].to_numpy(dtype=float)).all(), "Non-finite source data; no exclusion permitted")


def panel_letter(ax, letter, y_offset=12):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-30, y_offset),
                textcoords="offset points", ha="left", va="bottom", fontsize=8,
                fontweight="bold", annotation_clip=False)


def pair_scatter(ax, x, y, limits, xlabel, ylabel, letter, stats, title=None,
                 marker_size=8, marker_alpha=0.65):
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color=LIGHT_GRAY, lw=0.6, zorder=0)
    ax.axvline(0, color=LIGHT_GRAY, lw=0.6, zorder=0)
    ax.plot(limits, limits, color=GRAY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    # Every observation is shown. Transparency exposes overlapping points.
    ax.scatter(x, y, s=marker_size, c=TEAL, alpha=marker_alpha, linewidths=0, zorder=2)
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    # Descriptive summaries sit in reserved space above the plotting area;
    # no observations are obscured by the text.
    annotation = (
        f"Pearson r = {stats['pearson_r']:.3f}\n"
        f"Sign flips: {100 * stats['strict_sign_reversal_fraction']:.1f}% "
        f"({stats['strict_sign_reversal_count']}/{stats['n']})"
    )
    ax.annotate(annotation, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
                textcoords="offset points", ha="left", va="bottom", fontsize=8,
                linespacing=1.35, annotation_clip=False)
    if title:
        ax.set_title(title, pad=36)
    panel_letter(ax, letter, y_offset=36 if title else 12)


def global_limits(values):
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo
    return (lo - 0.07 * span, hi + 0.07 * span)


def save_figure(fig, stem, require_matplotlib_panel_alignment):
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig, json_out=validation_path(stem.parent, stem.name + ".alignment.json"),
        overlay_svg=validation_path(stem.parent, stem.name + ".alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=len(fig.axes) > 1, strict=True,
    )
    # Keep exact physical dimensions; do not crop with bbox_inches='tight'.
    fig.savefig(export_path(stem.parent, stem.name, ".svg"))
    fig.savefig(export_path(stem.parent, stem.name, ".pdf"))
    fig.savefig(export_path(stem.parent, stem.name, ".png"), dpi=300)
    fig.savefig(export_path(stem.parent, stem.name, ".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return alignment


def run_qa(skill, out, stems):
    source_check = subprocess.run(
        [sys.executable, str(skill / "scripts/validate_figure.py"), str(Path(__file__).resolve()), "--json"],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    validation_path(out, "supplement_source_validation.json").write_text(source_check.stdout, encoding="utf-8")
    require(source_check.returncode == 0, f"Source validation failed: {source_check.stdout} {source_check.stderr}")
    qa = {}
    for name in stems:
        stem = out / name
        text_check = subprocess.run(
            [sys.executable, str(skill / "scripts/audit_pdf_text.py"), str(export_path(stem.parent, stem.name, ".pdf")), "--min-pt", "5", "--json"],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        validation_path(stem.parent, stem.name + ".text-audit.json").write_text(text_check.stdout, encoding="utf-8")
        require(text_check.returncode == 0, f"PDF text QA failed for {name}: {text_check.stdout}")
        collision = subprocess.run(
            [sys.executable, str(skill / "scripts/audit_figure_collisions.py"), str(export_path(stem.parent, stem.name, ".pdf")),
             "--json-out", str(validation_path(stem.parent, stem.name + ".collision-audit.json")),
             "--overlay-pdf", str(validation_path(stem.parent, stem.name + ".collision-audit.pdf"))],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        require(collision.returncode == 0, f"PDF collision QA failed for {name}: {collision.stdout} {collision.stderr}")
        qa[name] = {
            "text_audit": json.loads(text_check.stdout),
            "collision_audit": json.loads(validation_path(stem.parent, stem.name + ".collision-audit.json").read_text(encoding="utf-8")),
            "alignment": json.loads(validation_path(stem.parent, stem.name + ".alignment.json").read_text(encoding="utf-8")),
        }
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run", default="20260908T224807Z")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skill-dir", type=Path, default=Path.home() / ".codex/skills/nature-figure")
    args = parser.parse_args()
    sys.path.insert(0, str(args.skill_dir / "scripts"))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    run = args.project / "results/formal" / args.run
    out = args.output or args.project / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_data").mkdir(parents=True, exist_ok=True)
    (out / "supplementary").mkdir(parents=True, exist_ok=True)
    inputs = {
        "direct_orders": run / "direct/direct_judgement_order.csv",
        "seed_distances": run / "distances/qwen3p5_9b_formal_triple_seed_distances.csv",
        "mean_distances": run / "distances/qwen3p5_9b_formal_sentence_distances.csv",
        "surprisal_items": run / "surprisal/shared_q3_surprisal_target_items.csv",
        "surprisal_summary": run / "surprisal/shared_q3_surprisal_summary.json",
    }
    hashes = {key: file_hash(path) for key, path in inputs.items()}
    direct = pd.read_csv(inputs["direct_orders"])
    require(len(direct) == 3520, "Expected all 3520 direct-order records")
    require(not direct.duplicated(["triple_id", "judgement_context", "candidate_order"]).any(), "Duplicate direct-order records")
    require(direct.groupby(["judgement_context", "candidate_order"]).size().eq(880).all(), "Incomplete candidate-order cells")
    require_finite(direct, ["apt_minus_inapt_nats", "log_probability_a_nats", "log_probability_b_nats"])
    direct["label_a_minus_b_nats"] = direct.log_probability_a_nats - direct.log_probability_b_nats
    mapped = np.where(direct.candidate_order == "AI", direct.label_a_minus_b_nats, -direct.label_a_minus_b_nats)
    require(np.allclose(mapped, direct.apt_minus_inapt_nats, atol=1e-10, rtol=0), "Candidate label remapping mismatch")
    order_items = direct.groupby(["sentence_id", "source_sid", "judgement_context", "candidate_order"], as_index=False).agg(
        apt_minus_inapt_nats=("apt_minus_inapt_nats", "mean"),
        label_a_minus_b_nats=("label_a_minus_b_nats", "mean"),
        triple_count=("triple_id", "size"),
    )
    require(len(order_items) == 2380, "Expected 595 items in all four cells")
    order_items.to_csv(out / "source_data" / "FigS1_candidate_order_source.csv", index=False)
    order_pairs = order_items.pivot(index=["sentence_id", "source_sid", "judgement_context"], columns="candidate_order", values="apt_minus_inapt_nats").reset_index()
    require_finite(order_pairs, ["AI", "IA"])
    limits_order = global_limits(order_pairs[["AI", "IA"]].to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM * MM, 110 * MM))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.23, top=0.77, wspace=0.35)
    order_stats = {}
    for ax, context, letter, title in zip(axes, ["context", "word"], "ab", ["With context", "Target word only"]):
        rows = order_pairs[order_pairs.judgement_context == context]
        require(len(rows) == 595, "Wrong item count for candidate-order panel")
        order_stats[context] = describe_pair(rows.AI, rows.IA)
        pair_scatter(ax, rows.AI, rows.IA, limits_order,
                     "AI-order preference (nats)", "IA-order preference (nats)",
                     letter, order_stats[context], title)
    fig.text(0.10, 0.055, "Strict sign flips: positive to negative or negative to positive; zero transitions not counted.",
             fontsize=8, ha="left", va="bottom")
    save_figure(fig, out / "FigS1_candidate_order", require_matplotlib_panel_alignment)

    seeds = pd.read_csv(inputs["seed_distances"])
    require(len(seeds) == 2640, "Expected all 2640 triple-seed records")
    require(not seeds.duplicated(["triple_id", "seed"]).any(), "Duplicate triple-seed records")
    require(set(seeds.seed) == {11, 23, 37}, "Unexpected generation seeds")
    require_finite(seeds, ["m_a_distance", "m_i_distance", "mi_minus_ma"])
    require(np.allclose(seeds.m_i_distance - seeds.m_a_distance, seeds.mi_minus_ma, atol=1e-10, rtol=0), "Distance contrast mismatch")
    seed_items = seeds.groupby(["sentence_id", "source_sid", "seed"], as_index=False).agg(
        delta=("mi_minus_ma", "mean"), m_a_distance=("m_a_distance", "mean"),
        m_i_distance=("m_i_distance", "mean"), triple_count=("triple_id", "size"),
    )
    require(len(seed_items) == 1785, "Expected 595 items for each seed")
    seed_items.to_csv(out / "source_data" / "FigS2_seed_comparison_source.csv", index=False)
    seed_pairs = seed_items.pivot(index=["sentence_id", "source_sid"], columns="seed", values="delta").reset_index()
    require(len(seed_pairs) == 595, "Wrong item count for seed comparisons")
    require_finite(seed_pairs, [11, 23, 37])
    require(set(seed_pairs.sentence_id) == set(order_pairs.sentence_id), "Items differ between seed and order figures")
    mean_distances = pd.read_csv(inputs["mean_distances"]).sort_values("sentence_id")
    reference_column = "mi_minus_ma" if "mi_minus_ma" in mean_distances else "mi_minus_ma_triple_mean"
    require(np.allclose(seed_pairs.sort_values("sentence_id")[[11, 23, 37]].mean(axis=1), mean_distances[reference_column], atol=1e-10, rtol=0), "Seed aggregation does not match existing item-level estimates")
    limits_seed = global_limits(seed_pairs[[11, 23, 37]].to_numpy())
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_MM * MM, 88 * MM))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.31, top=0.76, wspace=0.55)
    seed_stats = {}
    for ax, (first, second), letter in zip(axes, itertools.combinations([11, 23, 37], 2), "abc"):
        seed_stats[f"{first}_vs_{second}"] = describe_pair(seed_pairs[first], seed_pairs[second])
        pair_scatter(ax, seed_pairs[first], seed_pairs[second], limits_seed,
                     f"Δ, seed {first}", f"Δ, seed {second}", letter,
                     seed_stats[f"{first}_vs_{second}"], marker_size=5, marker_alpha=0.4)
    fig.text(0.085, 0.11, "Single-seed scores; main analyses use three-seed averages.",
             fontsize=8, ha="left", va="bottom")
    fig.text(0.085, 0.055, "Strict sign flips: positive to negative or negative to positive; zero transitions not counted.",
             fontsize=8, ha="left", va="bottom")
    save_figure(fig, out / "FigS2_seed_comparison", require_matplotlib_panel_alignment)

    surprisal = pd.read_csv(inputs["surprisal_items"])
    require(len(surprisal) == 595, "Expected all 595 surprisal items")
    require(surprisal.source_sid.nunique() == 553, "Expected 553 source clusters")
    require(set(surprisal.sentence_id) == set(seed_pairs.sentence_id), "Surprisal item identities do not match")
    require_finite(surprisal, ["inapt_minus_apt_nats_triple_mean"])
    surprisal.to_csv(out / "source_data" / "FigS3_surprisal_source.csv", index=False)
    coeff = json.loads(inputs["surprisal_summary"].read_text(encoding="utf-8"))["inference"]["coefficients"]["intercept"]
    values = surprisal.inapt_minus_apt_nats_triple_mean.to_numpy()
    require(np.isclose(np.mean(values), coeff["estimate"], atol=1e-10, rtol=0), "Surprisal mean mismatch")
    ci = coeff["confidence_interval_95"]
    fig, ax = plt.subplots(figsize=(89 * MM, 66 * MM))
    fig.subplots_adjust(left=0.18, right=0.965, bottom=0.23, top=0.92)
    counts, edges, _ = ax.hist(values, bins="fd", color=TEAL, edgecolor="white", linewidth=0.35, zorder=2)
    ax.axvline(0, color=GRAY, lw=0.8, zorder=3)
    ax.axvline(coeff["estimate"], color=DARK_TEAL, lw=1, ls=(0, (4, 3)), zorder=3)
    ax.set_ylim(0, float(max(counts)) * 1.08)
    ax.set_xlim(float(min(edges)), float(max(edges)))
    ax.set_xlabel("Surprisal contrast, F (nats)", labelpad=5)
    ax.set_ylabel("Target items", labelpad=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    save_figure(fig, out / "FigS3_surprisal", require_matplotlib_panel_alignment)

    captions = (
        "# Supplementary figure captions\n\n"
        "**Supplementary Fig. S1 | Candidate-order sensitivity of explicit preference.** "
        "Each point is one target item (n = 595), averaging all associated triples separately for each candidate order. "
        "Panels show judgements with context (a) and from the target word alone (b). "
        "AI places the apt candidate first; IA places it second. Both axes give the log probability of the apt response minus that of the inapt response after remapping the A/B response labels; positive values favour the apt candidate. "
        "The dashed diagonal marks identical scores across orders, and grey zero lines identify preference reversals. "
        "Both panels use the same scale. "
        f"For panels a and b, Pearson r = {order_stats['context']['pearson_r']:.3f} and {order_stats['word']['pearson_r']:.3f}; "
        f"strict positive-to-negative or negative-to-positive reversals occur in {order_stats['context']['strict_sign_reversal_count']}/595 and {order_stats['word']['strict_sign_reversal_count']}/595 items, respectively. "
        "Transitions involving zero are not counted as strict reversals. Source Data are provided.\n\n"
        "**Supplementary Fig. S2 | Agreement of item-level continuation alignment across generation seeds.** "
        "All three seed pairs are shown: 11 and 23 (a), 11 and 37 (b), and 23 and 37 (c). "
        "Each point is one of 595 target items. Within each seed, Δ is the mean across associated triples of the M–I distance minus the M–A distance; positive values indicate closer alignment with the apt replacement. "
        "Each condition contains 32 continuations per seed. The dashed diagonal denotes identical scores, and grey lines mark zero. "
        "All panels have equal axis scales. "
        f"For panels a, b and c, Pearson r = {seed_stats['11_vs_23']['pearson_r']:.3f}, {seed_stats['11_vs_37']['pearson_r']:.3f} and {seed_stats['23_vs_37']['pearson_r']:.3f}; "
        f"strict sign reversals occur in {seed_stats['11_vs_23']['strict_sign_reversal_count']}/595 ({100 * seed_stats['11_vs_23']['strict_sign_reversal_fraction']:.1f}%), "
        f"{seed_stats['11_vs_37']['strict_sign_reversal_count']}/595 ({100 * seed_stats['11_vs_37']['strict_sign_reversal_fraction']:.1f}%) and "
        f"{seed_stats['23_vs_37']['strict_sign_reversal_count']}/595 ({100 * seed_stats['23_vs_37']['strict_sign_reversal_fraction']:.1f}%) items. "
        "A strict reversal is any change between positive and negative values, regardless of magnitude. "
        "These comparisons use single-seed scores; the main analyses use three-seed averages. "
        "Source Data are provided for all 1,785 item-by-seed observations.\n\n"
        "**Supplementary Fig. S3 | Distribution of the shared three-word right-context surprisal contrast.** "
        "The histogram includes all 595 target items and uses Freedman–Diaconis bins. "
        "F is surprisal of the exact observed three-word span after the target under the inapt replacement minus surprisal of the same span under the apt replacement, summed over tokenizer tokens in that observed span and averaged across associated triples, in natural-log units. "
        "Positive values indicate greater surprisal after the inapt replacement. "
        f"The solid grey line marks zero and the dashed teal line marks the mean ({coeff['estimate']:.3f} nats; 95% confidence interval, {ci['lower']:.3f}–{ci['upper']:.3f}). "
        "The interval uses one-way CR1 cluster-robust standard errors across 553 source sentences with a Student t reference distribution (552 degrees of freedom). "
        "No observations are excluded. Source Data are provided.\n"
    )
    (out / "supplementary" / "captions.md").write_text(captions, encoding="utf-8")
    summary = {
        "model": "Qwen3.5-9B", "run": args.run,
        "counts": {"source_clusters": 553, "target_items": 595, "triples": 880, "direct_order_rows": 3520, "seed_rows": 2640},
        "excluded_observations": 0,
        "S1_axis_limits": limits_order, "S1": order_stats,
        "S2_axis_limits": limits_seed, "S2": seed_stats,
        "S3": {"mean": coeff["estimate"], "ci95": [ci["lower"], ci["upper"]], "positive": int(np.sum(values > 0)), "negative": int(np.sum(values < 0)), "histogram_bins": len(edges) - 1},
        "input_sha256": hashes,
    }
    write_json(out / "source_data" / "supplement_statistics.json", summary)
    qa = run_qa(args.skill_dir, out, ["FigS1_candidate_order", "FigS2_seed_comparison", "FigS3_surprisal"])
    require(hashes == {key: file_hash(path) for key, path in inputs.items()}, "An input file was modified")
    write_json(validation_path(out, "supplement_automated_qa.json"), qa)
    print(json.dumps(summary, indent=2))
    print("Supplementary figures exported; automated source, alignment, PDF text and collision QA passed.")


if __name__ == "__main__":
    main()
