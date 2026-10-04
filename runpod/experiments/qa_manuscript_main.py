#!/usr/bin/env python3
"""Preserve source and final-PDF QA for the revised main quantitative displays."""
import json
import os
import subprocess
import sys
from pathlib import Path
from figure_paths import export_path, validation_path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures'
QA=validation_path(OUT, 'main_qa_status.json').parent
SKILL=Path(os.environ.get('NATURE_FIGURE_SKILL',Path.home()/'.codex'/'skills'/'nature-figure'))/'scripts'

def run(tool,args,report):
    completed=subprocess.run([sys.executable,str(SKILL/tool),*map(str,args)],capture_output=True,text=True,encoding='utf-8')
    (QA/report).write_text(completed.stdout,encoding='utf-8')
    if completed.stderr:
        (QA/(report+'.stderr.txt')).write_text(completed.stderr,encoding='utf-8')
    return completed.returncode

def main():
    QA.mkdir(parents=True,exist_ok=True)
    status={}
    status['source']=run('validate_figure.py',[ROOT/'runpod/experiments'/'build_manuscript_figures.py','--json','--strict'],'main_source_validation.json')
    for name in ['fig2_rq1_distribution','fig3_overall_association','fig4_conditional_association','table1_materials','table2_rq2_models']:
        text_code=run('audit_pdf_text.py',[export_path(OUT, name, '.pdf'),'--min-pt','5','--json'],name+'.text.json')
        collision_code=run('audit_figure_collisions.py',[export_path(OUT, name, '.pdf'),'--json-out',QA/(name+'.collision.json'),'--overlay-pdf',QA/(name+'.collision.pdf')],name+'.collision.txt')
        status[name]={'text_exit':text_code,'collision_exit':collision_code}
    (QA/'main_qa_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))

if __name__=='__main__':
    main()
