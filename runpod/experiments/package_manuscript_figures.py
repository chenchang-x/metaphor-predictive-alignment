#!/usr/bin/env python3
"""Build the figure gallery, delivery checksums and reproduction archive."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import zipfile
from pathlib import Path

from figure_paths import export_path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures'
QA = ROOT / 'docs' / 'figure_validation'
DISPLAYS = [
    ('main', 'Figure 1', 'Experimental overview', 'fig1_experiment_overview', 'Real prefixes, two tasks and their shared item-level scores.'),
    ('main', 'Figure 2', 'Continuation alignment', 'fig2_rq1_distribution', 'All 595 items, with zero, the mean and its 95% confidence interval.'),
    ('main', 'Figure 3', 'Overall association', 'fig3_overall_association', 'Contextual preference and continuation alignment, with the formal model fit.'),
    ('main', 'Figure 4', 'Conditional association', 'fig4_conditional_association', 'Raw correlation −0.001 and adjusted partial correlation 0.146.'),
    ('supplementary', 'Figure S1', 'Candidate-order sensitivity', 'FigS1_candidate_order', 'Correlations and strict sign-reversal rates for both input conditions.'),
    ('supplementary', 'Figure S2', 'Item scores across seeds', 'FigS2_seed_comparison', 'Pairwise correlations and single-seed sign reversals of 27.7%–30.4%.'),
    ('supplementary', 'Figure S3', 'Local surprisal distribution', 'FigS3_surprisal', 'Auxiliary three-word surprisal contrast; its caption reports the mean and confidence interval.'),
    ('tables', 'Table 1', 'Material hierarchy and genres', 'table1_materials', '553 source sentences, 595 target items and 880 triples.'),
    ('tables', 'Table 2', 'Two RQ2 models', 'table2_rq2_models', 'All coefficients, standard errors, confidence intervals, t statistics and two-sided p values.'),
]


def visible_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob('*') if p.is_file()
                  and not any(part.startswith('.') for part in p.relative_to(directory).parts))


def write_gallery() -> Path:
    sections = []
    for group, heading in [('main', 'Main figures'), ('supplementary', 'Supplementary figures'), ('tables', 'Tables')]:
        sections.append(f'<section id="{group}"><h2>{heading}</h2>')
        for category, label, title, stem, note in DISPLAYS:
            if category != group:
                continue
            paths = {ext: export_path(OUT, stem, '.' + ext) for ext in ['pdf', 'png', 'svg', 'tiff']}
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            links = ' '.join(f'<a href="{path.relative_to(OUT).as_posix()}" download>{ext.upper()}</a>' for ext, path in paths.items())
            sections.append(f'<article><div class="label">{label}</div><h3>{html.escape(title)}</h3>'
                            f'<p>{html.escape(note)}</p><a href="{paths["pdf"].relative_to(OUT).as_posix()}">'
                            f'<img src="{paths["png"].relative_to(OUT).as_posix()}" alt="{html.escape(title)}"></a><nav>{links}</nav></article>')
        sections.append('</section>')
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.5-9B | Figures and Tables</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f7f6;color:#263c3d;font:16px/1.7 system-ui,sans-serif}
main{max-width:1060px;margin:auto;padding:40px 24px}h1{font-size:32px;line-height:1.3}h2{margin:52px 0 20px}h3{font-size:22px;margin:4px 0}
article{background:white;padding:28px 32px;margin:24px 0;border:1px solid #e0e7e4;border-radius:8px}
img{display:block;max-width:100%;height:auto;margin:24px auto}nav,.links{display:flex;flex-wrap:wrap;gap:16px}a{color:#267f7b;text-underline-offset:4px}
.label{color:#667c7a;font-size:14px}.links{padding:18px;background:#e9f1ee;border-radius:6px;margin-bottom:18px}
@media print{body{background:white}main{padding:0}article{border:0;break-after:page}nav,.links{display:none}}
</style></head><body><main><header><div class="label">Formal run 20260908T224807Z</div>
<h1>Qwen3.5-9B Figures and Tables</h1><p>Four main figures, three supplementary figures and two tables. PDF is the primary viewing format; PNG, SVG and TIFF exports remain available.</p>
<nav class="links"><a href="#main">Main figures</a><a href="#supplementary">Supplementary figures</a><a href="#tables">Tables</a></nav>
<div class="links"><a href="main/captions.md">Main captions</a><a href="supplementary/captions.md">Supplementary captions</a><a href="tables/tables.md">Copyable tables</a><a href="source_data/">Source data</a><a href="README.md">Reproduction</a><a href="../docs/formal_20260908T224807Z_summary.md">Results</a><a href="../docs/figure_validation/README.md">Validation records</a></div></header>
'''
    page += '\n'.join(sections) + '</main></body></html>\n'
    for target in re.findall(r'(?:href|src)="([^"]+)"', page):
        if not target.startswith('#') and not (OUT / target).exists():
            raise FileNotFoundError(target)
    path = OUT / 'index.html'
    path.write_text(page, encoding='utf-8')
    return path


def write_manifest() -> Path:
    QA.mkdir(parents=True, exist_ok=True)
    path = QA / 'delivery_manifest.json'
    files = [p for p in visible_files(OUT) + visible_files(QA) if p != path]
    manifest = {'model': 'Qwen/Qwen3.5-9B', 'run': '20260908T224807Z',
                'main_figures': 4, 'main_tables': 2, 'supplement_figures': 3,
                'path_base': 'repository root',
                'files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


def build_archive(path: Path) -> int:
    files = set(visible_files(OUT) + visible_files(QA))
    files.update((ROOT / 'runpod' / 'experiments').glob('*.py'))
    files.update((ROOT / 'src' / 'qwen_alignment').glob('*.py'))
    for extension in ['*.csv', '*.json']:
        files.update((ROOT / 'vendor' / 'metaphor-understanding-challenge').rglob(extension))
        files.update((ROOT / 'results' / 'formal' / '20260908T224807Z').rglob(extension))
    files.update(ROOT / relative for relative in ['data/processed/analysis_items.csv', 'docs/formal_20260908T224807Z_summary.md',
        'docs/protocol.md', 'docs/protocol_en.md', 'environment/tokenization_audit_summary.json'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in sorted(files):
            if not item.is_file():
                raise FileNotFoundError(item)
            archive.write(item, 'qwen3p5_figures_20260911/' + item.relative_to(ROOT).as_posix())
        archive.writestr('qwen3p5_figures_20260911/OPEN_FIRST.txt',
                         'Open figures/index.html to browse the figures.\nSee figures/README.md for reproduction commands.\n'
                         'Saved experimental evidence is included; no model weights are needed for plotting.\n')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
    return len(files) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery-only', action='store_true', help='Update the gallery and checksums without building an archive.')
    parser.add_argument('--archive', type=Path, default=ROOT / 'transfer' / 'manuscript_20260911.zip')
    args = parser.parse_args()
    gallery = write_gallery()
    manifest = write_manifest()
    report = {'gallery': str(gallery), 'manifest': str(manifest), 'display_objects': len(DISPLAYS)}
    if not args.gallery_only:
        report['archive_entries'] = build_archive(args.archive)
        report['archive'] = str(args.archive)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
