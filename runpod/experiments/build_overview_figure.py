"""Draw the source-grounded two-task experimental overview with editable vectors.

Figure contract: show how the same target items yield continuation alignment
and explicit preferences, and how their scores enter the two RQ2 models.
Archetype: schematic-led composite, with two complementary task branches.
All example strings are read from the frozen processed and formal-run records.
No generated continuation is invented or used as a representative result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from figure_paths import export_path, validation_path
import subprocess
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
OUT = ROOT / "figures"
EXAMPLE_ID = "munch-judgement-1232"
WIDTH_MM = 183
HEIGHT_MM = 130
INK, GRAY, TEAL, PALE_TEAL = "#22282B", "#60686D", "#267F7B", "#5EA8A1"
LAVENDER = "#D7C3E8"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.facecolor": "white",
})


def text(ax, x, y, label, *, size=8.5, color=INK, weight="normal", **kwargs):
    return ax.text(x, y, label, fontsize=size, color=color, fontweight=weight,
                   va="center", clip_on=False, **kwargs)


def arrow(fig, start, end, color=GRAY):
    fig.patches.append(FancyArrowPatch(
        (start[0] / WIDTH_MM, start[1] / HEIGHT_MM),
        (end[0] / WIDTH_MM, end[1] / HEIGHT_MM),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
        linewidth=0.8, color=color, shrinkA=0, shrinkB=0,
    ))


def fig_text(fig, x, y, label, **kwargs):
    return fig.text(x / WIDTH_MM, y / HEIGHT_MM, label, va="center",
                    color=INK, fontsize=kwargs.pop("fontsize", 8.5), **kwargs)


def line(fig, xs, ys, color="#C6CDCF", width=0.7):
    fig.lines.append(Line2D([x / WIDTH_MM for x in xs],
                            [y / HEIGHT_MM for y in ys],
                            transform=fig.transFigure, color=color, lw=width))


def audit_tool(script: str, args: list[str], destination: Path) -> int:
    result = subprocess.run([sys.executable, str(SKILL_SCRIPTS / script), *args],
                            text=True, capture_output=True, encoding="utf-8")
    destination.write_text(result.stdout + ("\n" + result.stderr if result.stderr else ""),
                           encoding="utf-8")
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
    return result.returncode


def build() -> None:
    qa = validation_path(OUT, "fig1_contract.json").parent
    source_dir = OUT / "source_data"
    qa.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    materials = pd.read_csv(ROOT / "data/processed/analysis_items.csv")
    direct = pd.read_csv(ROOT / "results/formal/20260908T224807Z/direct/direct_judgement_order.csv")
    assert len(materials) == 880
    assert materials.sentence_id.nunique() == 595
    assert materials.source_sid.nunique() == 553
    example = materials.loc[materials.triple_id.eq(EXAMPLE_ID)].iloc[0]
    orders = direct.loc[direct.triple_id.eq(EXAMPLE_ID)]
    assert len(orders) == 4
    inputs = {}
    for condition in ("context", "word"):
        selected = orders.loc[orders.judgement_context.eq(condition)]
        assert set(selected.candidate_order) == {"AI", "IA"}
        assert selected.task_input.nunique() == 1
        inputs[condition] = selected.iloc[0].task_input
        for row in selected.itertuples():
            expected = ((example.a_word, example.i_word) if row.candidate_order == "AI"
                        else (example.i_word, example.a_word))
            assert (row.option_a, row.option_b) == expected
    assert inputs["context"].replace(f"[TARGET: {example.m_word}]", example.m_word) == example.m_prefix_q3
    assert inputs["word"] == f"[TARGET: {example.m_word}]"
    assert example.m_prefix_q3 != example.m_sentence
    assert example.q3_right_context.strip() == "the bus back"

    source_rows = []
    for role in ("m", "a", "i"):
        source_rows.append({"triple_id": EXAMPLE_ID, "sentence_id": example.sentence_id,
                            "source_sid": example.source_sid, "task": "continuation",
                            "condition": role.upper(), "candidate_order": "",
                            "task_input": example[f"{role}_prefix_q3"],
                            "option_a": "", "option_b": ""})
    for row in orders.itertuples():
        source_rows.append({"triple_id": EXAMPLE_ID, "sentence_id": example.sentence_id,
                            "source_sid": example.source_sid, "task": "explicit judgement",
                            "condition": row.judgement_context,
                            "candidate_order": row.candidate_order, "task_input": row.task_input,
                            "option_a": row.option_a, "option_b": row.option_b})
    pd.DataFrame(source_rows).to_csv(source_dir / "fig1_input.csv", index=False)

    contract = {
        "figure": "Fig. 1", "claim": "The same target items connect two separate tasks through item-level scores.",
        "roles": {"a": "Real M/A/I prefixes define continuation alignment.",
                  "b": "Matched context/word inputs define explicit preference and its context-linked change.",
                  "footer": "Two prespecified RQ2 models join scores for the same target items."},
        "archetype": "schematic-led composite", "backend": "python-matplotlib",
        "width_mm": WIDTH_MM, "height_mm": HEIGHT_MM, "font": "Arial",
        "body_minimum_pt": 8, "glyph_minimum_pt": 5, "example_id": EXAMPLE_ID,
        "example_selection": "A short, readable retained MUNCH item; selected without reference to outcome scores.",
        "input_integrity": "Exact real prefixes ending after the three shared post-target words; not the full sentence.",
        "generated_text": "No sampled or invented continuation text is shown.",
        "analysis_units": {"items": 595, "source_clusters": 553, "triple_records": 880},
        "uncertainty": "Not applicable to this experimental schematic; no empirical effect estimate is drawn.",
        "exports": ["editable SVG", "PDF", "PNG 300 dpi", "TIFF 600 dpi"],
    }
    (qa / "fig1_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, left=8 / WIDTH_MM, right=175 / WIDTH_MM,
                          bottom=28 / HEIGHT_MM, top=110 / HEIGHT_MM, wspace=0.16)
    left, right = [fig.add_subplot(gs[0, j]) for j in (0, 1)]
    for ax in (left, right):
        ax.set_xlim(0, 78)
        ax.set_ylim(28, 110)
        ax.set_axis_off()

    fig_text(fig, 8, 123, "Two tasks, one item-level comparison", fontsize=12, fontweight="bold")
    fig.text(8 / WIDTH_MM, 116.5 / HEIGHT_MM, "Qwen3.5-9B  ·  595 target items",
             fontsize=9, va="center", color=GRAY)

    text(left, 0, 106, "a", size=10, weight="bold")
    text(left, 5, 106, "Continuation", size=10, weight="bold", color=TEAL)
    text(left, 0, 98, "Three matched contextual prefixes", size=8.5, color=GRAY)
    for y, role, color in ((89, "M", GRAY), (81, "A", PALE_TEAL), (73, "I", LAVENDER)):
        left.plot([0, 1.3], [y, y], color=color, linewidth=3, solid_capstyle="butt")
        text(left, 3.4, y, role, size=9, weight="bold")
        text(left, 10, y, example[f"{role.lower()}_prefix_q3"], size=9)
    text(left, 0, 66, "M: original  ·  A: apt  ·  I: inapt", size=8, color=GRAY)
    text(left, 0, 58.5, "Prefix ends after 3 shared post-target words", size=8, color=GRAY)
    arrow(fig, (45, 54), (45, 50.5), TEAL)
    text(left, 0, 46, "Sample short continuations", size=9, weight="bold")
    text(left, 0, 40.5, "32 samples × 3 seeds  ·  5 tokens each", size=8)
    text(left, 0, 33.5, r"$\Delta_i = d_i(M,I) - d_i(M,A)$", size=10, color=TEAL)
    text(left, 0, 28.5, "Positive Δ: metaphor continuations are closer to A", size=8)

    text(right, 0, 106, "b", size=10, weight="bold")
    text(right, 5, 106, "Explicit judgement", size=10, weight="bold", color=TEAL)
    text(right, 0, 98, "Matched context", size=8.5, color=GRAY)
    text(right, 0, 91, inputs["context"], size=9)
    text(right, 0, 83, "Word only", size=8.5, color=GRAY)
    text(right, 0, 76, inputs["word"], size=9)
    text(right, 0, 67, f"Candidates: {example.a_word} (apt) / {example.i_word} (inapt)", size=8.5)
    text(right, 0, 61, "Same question and A/B responses; both option orders", size=8)
    arrow(fig, (135, 56.5), (135, 53), TEAL)
    text(right, 0, 48.5, "Apt-minus-inapt log response probability", size=8.5)
    text(right, 0, 42, r"$S_{\mathrm{context},i}$     and     $S_{\mathrm{word},i}$", size=10)
    text(right, 0, 34.5, r"$C_i = S_{\mathrm{context},i} - S_{\mathrm{word},i}$", size=10, color=TEAL)
    text(right, 0, 28.5, "Context-linked change in explicit preference", size=8)

    # The two branches meet once: the same target-item scores enter RQ2.
    line(fig, [46, 46, 137, 137], [25, 22.5, 22.5, 25], color=TEAL, width=0.8)
    arrow(fig, (91.5, 22.5), (91.5, 20.5), TEAL)
    fig_text(fig, 91.5, 17, "RQ2 · Combine scores from the same 595 target items",
             fontsize=9.5, fontweight="bold", ha="center")
    fig_text(fig, 8, 10.5, "M0  Overall association", fontsize=8.5, fontweight="bold")
    fig_text(fig, 8, 5, r"$\Delta$ with contextual preference $S_{\mathrm{context}}$", fontsize=8.5)
    fig_text(fig, 98, 10.5, "M1  Conditional association", fontsize=8.5, fontweight="bold")
    fig_text(fig, 98, 5, r"$\Delta$ with $C$, accounting for $S_{\mathrm{word}}$", fontsize=8.5)

    spec = importlib.util.spec_from_file_location("audit_panel_alignment", SKILL_SCRIPTS / "audit_panel_alignment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fig.canvas.draw()
    module.require_matplotlib_panel_alignment(
        fig, json_out=qa / "fig1_alignment.json", overlay_svg=qa / "fig1_alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    stem = OUT / "fig1_experiment_overview"
    fig.savefig(export_path(OUT, stem.name, ".svg"))
    fig.savefig(export_path(OUT, stem.name, ".pdf"))
    fig.savefig(export_path(OUT, stem.name, ".png"), dpi=300)
    fig.savefig(export_path(OUT, stem.name, ".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    checks = [
        audit_tool("validate_figure.py", [str(Path(__file__)), "--json"], qa / "fig1_source_validation.json"),
        audit_tool("audit_pdf_text.py", [str(export_path(OUT, stem.name, ".pdf")), "--json"], qa / "fig1_pdf_text_audit.json"),
        audit_tool("audit_figure_collisions.py", [str(export_path(OUT, stem.name, ".pdf")),
                   "--json-out", str(qa / "fig1_collision_audit.json"),
                   "--overlay-pdf", str(qa / "fig1_collision_audit.pdf")], qa / "fig1_collision_audit.txt"),
    ]
    caption = "# Figure 1 | Connecting the two tasks at the item level\n\nQwen3.5-9B performs continuation generation and explicit judgement. In panel a, a retained item pairs the original expression caught (M), the apt interpretation took (A), and the inapt interpretation grabbed (I). Each input ends after the shared three post-target words, the bus back. These are contextual prefixes rather than complete source sentences. Each condition yields 32 continuations of 5 model tokens per generation seed. Set distance is the mean pairwise cosine distance between separately encoded five-token continuations, standardised using a shared reference; computation details appear in Methods. Distances are aggregated into the item-level contrast Δᵢ = dᵢ(M,I) − dᵢ(M,A). Positive values indicate that metaphor-condition continuations are closer to the apt-interpretation condition.\n\nIn panel b, explicit judgement presents either the same target-marked contextual prefix or the target word alone. Candidates, questions and A/B responses are matched, and candidate positions are reversed. Apt-minus-inapt log response probabilities yield S_context,ᵢ and S_word,ᵢ. Their difference, Cᵢ, measures the preference change after adding context. The two tasks are joined for the same 595 target items. M0 relates Δᵢ to S_context,ᵢ; M1 relates Δᵢ to Cᵢ while including S_word,ᵢ.\n\nThe figure shows inputs and calculation relationships, not sampled continuation examples. Continuation contrasts are averaged across seeds and then across triples within each target item. Explicit scores are balanced across candidate orders before item-level aggregation. M/A/I label semantic conditions; A/B label candidate positions. Cᵢ is a measured context increment and does not imply the removal of every lexical influence.\n\nThe complete source sentence is “We caught the bus back to Ellen's apartment and collected her clothes and notebooks.” The figure shows the truncated prefixes used in the formal run. The example was chosen for readability, independently of outcome scores.\n"
    validation_path(OUT, "overview_caption.md").write_text(caption, encoding="utf-8")
    if any(checks):
        raise SystemExit("One or more figure QA checks require correction.")
    print(f"Wrote {stem.name}: SVG, PDF, PNG (300 dpi), TIFF (600 dpi); automatic QA completed.")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
