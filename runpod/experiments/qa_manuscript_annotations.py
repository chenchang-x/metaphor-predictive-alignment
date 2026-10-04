"""Make a 180-mm print proof and audit the final annotation revision."""
from pathlib import Path
from figure_paths import export_path, validation_path
import json
import subprocess
import sys
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures'
QA = validation_path(OUT, 'annotation_revision_qa.json').parent
SKILL = Path.home() / '.codex/skills/nature-figure/scripts'
MM = 72 / 25.4
NAMES = ['fig4_conditional_association', 'FigS1_candidate_order', 'FigS2_seed_comparison']


def main():
    proof = fitz.open()
    placements = []
    for page_names in [NAMES[:1], NAMES[1:]]:
        page = proof.new_page(width=210*MM, height=297*MM)
        page.insert_text((15*MM, 13*MM), 'Double-column proof | 180 mm wide | Print at 100%', fontsize=9)
        top = 23*MM
        for name in page_names:
            with fitz.open(export_path(OUT, name, '.pdf')) as source:
                source_page = source[0]
                width = 180*MM
                height = width * source_page.rect.height / source_page.rect.width
                rect = fitz.Rect(15*MM, top, 15*MM+width, top+height)
                assert rect.y1 < 278*MM
                page.show_pdf_page(rect, source, 0)
                placements.append({'figure': name, 'proof_width_mm': 180,
                                   'proof_height_mm': height/MM, 'scale': width/source_page.rect.width})
                top += height + 14*MM
        page.insert_text((15*MM, 287*MM), 'Every scatter panel retains all 595 items. Native figure width: 183 mm.', fontsize=8)
    target = QA / 'double_column_proof_180mm.pdf'
    proof.save(target)
    proof.close()
    scaled_sizes = []
    with fitz.open(target) as rendered:
        for i,page in enumerate(rendered):
            page.get_pixmap(dpi=150, alpha=False).save(QA / f'double_column_proof_180mm_page{i+1}.png')
            scaled_sizes.extend(span['size'] for block in page.get_text('dict')['blocks'] if block['type']==0
                                for line in block['lines'] for span in line['spans'] if span['text'].strip())
    assert scaled_sizes and min(scaled_sizes) >= 5, scaled_sizes
    text_run = subprocess.run([sys.executable, str(SKILL/'audit_pdf_text.py'), str(target), '--min-pt', '5', '--json'],
                              capture_output=True, text=True, encoding='utf-8')
    (QA/'double_column_proof_180mm.text.json').write_text(text_run.stdout, encoding='utf-8')
    assert text_run.returncode == 0, text_run.stdout + text_run.stderr
    collision_run = subprocess.run([sys.executable, str(SKILL/'audit_figure_collisions.py'), str(target),
                                   '--json-out', str(QA/'double_column_proof_180mm.collision.json'),
                                   '--overlay-pdf', str(QA/'double_column_proof_180mm.collision.pdf')],
                                  capture_output=True, text=True, encoding='utf-8')
    assert collision_run.returncode == 0, collision_run.stdout + collision_run.stderr
    annotation_text = {}
    for name in NAMES:
        with fitz.open(export_path(OUT, name, '.pdf')) as doc:
            annotation_text[name] = ' '.join(doc[0].get_text().split())
    assert 'Pearson r = −0.001' in annotation_text[NAMES[0]]
    assert 'Partial Pearson r = 0.146' in annotation_text[NAMES[0]]
    for val in ['0.831','0.843','9.7%','17.0%','58/595','101/595']:
        assert val in annotation_text[NAMES[1]], val
    for val in ['0.718','0.693','0.697','27.7%','30.4%','29.2%','165/595','181/595','174/595']:
        assert val in annotation_text[NAMES[2]], val
    report = {'status': 'PASS', 'annotation_strings_match_final_pdfs': True,
              'print_proof_placements': placements,
              'minimum_scaled_glyph_pt': min(scaled_sizes),
              'raw_font_operand_minimum_pt': json.loads(text_run.stdout)['minimum_found_pt'],
              'collision_summary': json.loads((QA/'double_column_proof_180mm.collision.json').read_text(encoding='utf-8'))['summary'],
              'interpretation': 'Descriptive Pearson and partial Pearson r; strict sign-flip numerators exclude zero transitions; all denominators remain 595.'}
    (QA/'annotation_revision_qa.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
