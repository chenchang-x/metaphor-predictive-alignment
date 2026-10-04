#!/usr/bin/env python3
"""Reproduce the main quantitative figures and compact tables from the formal run.

No model is loaded. Frozen M0/M1 estimates and covariance matrices are checked
against every retained observation; supplementary descriptive fits are labeled.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from figure_paths import export_path, validation_path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures'
RUN = ROOT / 'results' / 'formal' / '20260908T224807Z'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

SKILL = Path(os.environ.get('NATURE_FIGURE_SKILL', Path.home() / '.codex' / 'skills' / 'nature-figure'))
sys.path.insert(0, str(SKILL / 'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.65, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'text.color': '#293C3E', 'axes.labelcolor': '#293C3E',
    'xtick.color': '#526164', 'ytick.color': '#526164',
    'legend.frameon': False, 'savefig.facecolor': 'white',
})
TEAL = '#5EA8A1'
DARK = '#267F7B'
LILAC = '#D7C3E8'
GREY = '#879597'
width_mm = 183
TC = 1.9642708556448771


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def csv(frame, name):
    target = OUT / 'source_data' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, float_format='%.17g')


def save(fig, name, axes=None):
    fig.canvas.draw()
    if axes is not None and len(axes) > 1:
        require_matplotlib_panel_alignment(fig, axes=axes,
            json_out=validation_path(OUT, f'{name}.alignment.json'),
            overlay_svg=validation_path(OUT, f'{name}.alignment.svg'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    else:
        write_json(validation_path(OUT, f'{name}.alignment.json'),
                   {'status': 'not applicable', 'reason': 'Single plot area or compact table'})
    fig.savefig(export_path(OUT, name, '.pdf'))
    fig.savefig(export_path(OUT, name, '.svg'))
    fig.savefig(export_path(OUT, name, '.png'), dpi=300)
    fig.savefig(export_path(OUT, name, '.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def ols_cr1(X, y, clusters):
    """Independent numerical check using all N items and G source clusters."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    groups = np.unique(clusters)
    scores = np.array([X[clusters == g].T @ residual[clusters == g] for g in groups])
    n, k = X.shape
    correction = len(groups) / (len(groups) - 1) * (n - 1) / (n - k)
    covariance = correction * bread @ (scores.T @ scores) @ bread
    return beta, covariance


def limits(values, fraction=.06):
    low, high = float(np.min(values)), float(np.max(values))
    pad = (high - low) * fraction
    return low - pad, high + pad


def scatter(ax, x, y):
    ax.axhline(0, color=GREY, lw=.65, ls=(0, (3, 3)), zorder=0)
    ax.axvline(0, color=GREY, lw=.65, ls=(0, (3, 3)), zorder=0)
    ax.scatter(x, y, s=12, color=TEAL, alpha=.56, edgecolors='none', zorder=2)
    ax.set_xlim(limits(x))
    ax.set_ylim(limits(y))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def fit_band(ax, grid, X, beta, covariance):
    fitted = X @ beta
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', X, covariance, X), 0))
    ax.fill_between(grid, fitted - TC*se, fitted + TC*se,
                    color=TEAL, alpha=.19, linewidth=0, zorder=1)
    ax.plot(grid, fitted, color=DARK, lw=1.5, zorder=3)
    return fitted, se


def table_canvas(name, title, headers, rows, col_widths, notes, height):
    fig, ax = plt.subplots(figsize=(183/25.4, height/25.4))
    ax.set_axis_off()
    ax.set_position([.035, .26, .93, .53])
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='right',
                     colLoc='right', colWidths=col_widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = ''
        cell.PAD = .055
        if c == 0:
            cell.set_text_props(ha='left')
        if r == 0:
            cell.visible_edges = 'TB'
            cell.set_edgecolor('#637678')
            cell.set_linewidth(.7)
            cell.set_text_props(weight='bold')
        if r == len(rows):
            cell.visible_edges = 'B'
            cell.set_edgecolor('#637678')
            cell.set_linewidth(.7)
    fig.text(.035, .91, title, fontsize=10, weight='bold', va='top')
    fig.text(.035, .18, notes, fontsize=8, linespacing=1.6, va='top')
    save(fig, name)


def main():
    data_path = RUN / 'rq2' / 'rq2_items.csv'
    df = pd.read_csv(data_path)
    assert len(df) == 595 and df.sentence_id.nunique() == 595 and df.source_sid.nunique() == 553
    cols = ['p_context_i_nats', 'p_word_i_nats', 'c_i_nats', 'delta_i']
    assert np.isfinite(df[cols].to_numpy()).all()
    assert np.allclose(df.c_i_nats, df.p_context_i_nats - df.p_word_i_nats, atol=1e-12, rtol=0)
    y = df.delta_i.to_numpy()
    context = df.p_context_i_nats.to_numpy()
    word = df.p_word_i_nats.to_numpy()
    c = df.c_i_nats.to_numpy()
    groups = df.source_sid.to_numpy()
    primary = js(RUN / 'primary' / 'primary_analysis_results.json')['primary_effect']
    models = js(RUN / 'rq2' / 'rq2_results.json')['models']
    m0, m1 = models['M0_total_association'], models['M1_context_decomposition']
    X0 = np.column_stack([np.ones(len(df)), context])
    X1 = np.column_stack([np.ones(len(df)), c, word])
    b0 = np.array([m0['coefficients'][k]['estimate'] for k in ('beta_0','beta_total')])
    b1 = np.array([m1['coefficients'][k]['estimate'] for k in ('gamma_0','gamma_C','gamma_W')])
    v0, v1 = np.array(m0['covariance_matrix']), np.array(m1['covariance_matrix'])
    numerical = {}
    for name, X, b, v in [('M0',X0,b0,v0),('M1',X1,b1,v1)]:
        calc_b, calc_v = ols_cr1(X, y, groups)
        assert np.allclose(calc_b,b,atol=1e-12,rtol=1e-9)
        assert np.allclose(calc_v,v,atol=1e-14,rtol=1e-8)
        numerical[name] = {'max_coefficient_error':float(np.max(np.abs(calc_b-b))),
                           'max_covariance_error':float(np.max(np.abs(calc_v-v)))}
    mean_b, mean_v = ols_cr1(np.ones((len(df),1)), y, groups)
    assert abs(mean_b[0] - primary['estimate']) < 1e-12
    assert abs(mean_v[0,0] - primary['cr1_variance']) < 1e-14
    distances = pd.read_csv(RUN / 'distances' / 'qwen3p5_9b_formal_sentence_distances.csv')
    distance_col = 'mi_minus_ma_triple_mean'
    check = df.merge(distances[['sentence_id', distance_col]], on='sentence_id', validate='one_to_one')
    assert np.allclose(check.delta_i, check[distance_col], rtol=0, atol=1e-12)

    # Figure 2: equal-width bins have zero as an exact edge and include both tails.
    edges = np.arange(-12, 23, dtype=float) * .01
    counts, _ = np.histogram(y, edges)
    assert counts.sum() == 595
    fig, ax = plt.subplots(figsize=(120/25.4, 94/25.4))
    fig.subplots_adjust(left=.145, right=.965, bottom=.20, top=.78)
    ax.bar(edges[:-1], counts, width=.01, align='edge',
           color=[LILAC if e < 0 else TEAL for e in edges[:-1]],
           edgecolor='white', linewidth=.4)
    ymax = float(max(counts))*1.22
    ax.set_ylim(0, ymax)
    ax.set_xlim(edges[0], edges[-1])
    ax.axvline(0, color='#647375', lw=.9, ls=(0,(3,3)))
    mean = primary['estimate']
    ci = primary['confidence_interval_95_two_sided']
    ax.errorbar(mean, ymax*.92,
                xerr=np.array([[mean-ci['lower']], [ci['upper']-mean]]),
                fmt='o', color=DARK, ms=4, capsize=3, elinewidth=1.3, zorder=4)
    ax.set_xlabel(r'Continuation alignment, $\Delta_i=d_i(M,I)-d_i(M,A)$', labelpad=7)
    ax.set_ylabel('Target items (count)')
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.set_xticks([-.10,0,.10,.20])
    fig.text(.145,.94,'Mean = 0.0143  |  95% CI [0.0118, 0.0168]', fontsize=9, va='top')
    fig.text(.145,.865,'182 negative  ·  413 positive  ·  595 items',fontsize=8,va='top')
    fig.text(.145,.055,'Positive values: M continuations are closer to A than I.',fontsize=8)
    save(fig,'fig2_rq1_distribution')
    csv(df[['sentence_id','source_sid','delta_i']].rename(columns={'delta_i':'Delta_i'}), 'fig2_items.csv')
    csv(pd.DataFrame({'bin_left':edges[:-1], 'bin_right':edges[1:], 'count':counts}), 'fig2_histogram_bins.csv')

    # Figure 3: retain raw observations and the frozen full fitted-mean uncertainty.
    fig, ax = plt.subplots(figsize=(120/25.4, 100/25.4))
    fig.subplots_adjust(left=.155,right=.965,bottom=.20,top=.87)
    scatter(ax,context,y)
    grid=np.linspace(context.min(),context.max(),201)
    Xgrid=np.column_stack([np.ones(len(grid)),grid])
    fitted,se=fit_band(ax,grid,Xgrid,b0,v0)
    ax.set_xlabel(r'Contextual preference, $S_{\mathrm{context},i}$ (nats)',labelpad=7)
    ax.set_ylabel(r'Continuation alignment, $\Delta_i$')
    fig.text(.155,.95,'595 items',fontsize=8,va='top')
    fig.text(.965,.95,r'$R^2$ = 4.02%',fontsize=8,va='top',ha='right')
    fig.text(.155,.052,'OLS fit and pointwise 95% confidence band.',fontsize=8)
    save(fig,'fig3_overall_association')
    csv(df[['sentence_id','source_sid','p_context_i_nats','delta_i']].rename(columns={'p_context_i_nats':'S_context_i_nats','delta_i':'Delta_i'}),'fig3_items.csv')
    csv(pd.DataFrame({'S_context_i_nats':grid,'fitted_Delta':fitted,'ci_lower':fitted-TC*se,'ci_upper':fitted+TC*se}),'fig3_fit_band.csv')

    # Figure 4: Frisch-Waugh-Lovell residuals remove the same nuisance from BOTH axes.
    Z=np.column_stack([np.ones(len(df)),word])
    c_projection=np.linalg.lstsq(Z,c,rcond=None)[0]
    y_projection=np.linalg.lstsq(Z,y,rcond=None)[0]
    cr=c-Z@c_projection
    yr=y-Z@y_projection
    residual_slope=float(cr@yr/(cr@cr))
    assert abs(residual_slope-b1[1]) < 1e-12
    assert np.max(np.abs(Z.T@cr)) < 1e-9 and np.max(np.abs(Z.T@yr)) < 1e-9
    raw_r=float(np.corrcoef(c,y)[0,1])
    partial_r=float(np.corrcoef(cr,yr)[0,1])
    c_word_r=float(np.corrcoef(c,word)[0,1])
    y_word_r=float(np.corrcoef(y,word)[0,1])
    partial_r_formula=(raw_r-c_word_r*y_word_r)/np.sqrt((1-c_word_r**2)*(1-y_word_r**2))
    assert abs(partial_r-partial_r_formula) < 1e-12
    raw_X=np.column_stack([np.ones(len(df)),c])
    raw_b,raw_v=ols_cr1(raw_X,y,groups)
    fig,axs=plt.subplots(1,2,figsize=(183/25.4,100/25.4))
    fig.subplots_adjust(left=.10,right=.985,bottom=.235,top=.83,wspace=.38)
    for ax,x,yy,title,letter in zip(axs,[c,cr],[y,yr],['Unadjusted','Adjusted for word-only preference'],['a','b']):
        scatter(ax,x,yy)
        ax.text(-.17,1.14,letter,transform=ax.transAxes,fontsize=10,weight='bold')
        ax.set_title(title,pad=24,loc='left',fontsize=9)
    for ax,label,value in zip(axs,['Pearson r','Partial Pearson r'],[raw_r,partial_r]):
        display=f'{value:.3f}'.replace('-', '−')
        ax.text(0,1.025,f'{label} = {display}',transform=ax.transAxes,
                fontsize=8,va='bottom',ha='left',clip_on=False)
    shared_y=limits(np.r_[y,yr])
    shared_x=limits(np.r_[c,cr])
    for ax in axs:
        ax.set_ylim(shared_y)
        ax.set_xlim(shared_x)
    grid_a=np.linspace(c.min(),c.max(),201)
    raw_fit,raw_se=fit_band(axs[0],grid_a,np.column_stack([np.ones(len(grid_a)),grid_a]),raw_b,raw_v)
    grid_b=np.unique(np.r_[np.linspace(cr.min(),cr.max(),201),0.0])
    cond_fit=b1[1]*grid_b
    cond_se=np.abs(grid_b)*np.sqrt(v1[1,1])
    axs[1].fill_between(grid_b,cond_fit-TC*cond_se,cond_fit+TC*cond_se,color=TEAL,alpha=.19,linewidth=0,zorder=1)
    axs[1].plot(grid_b,cond_fit,color=DARK,lw=1.5,zorder=3)
    axs[0].set_xlabel(r'Context increment, $C_i$ (nats)',labelpad=7)
    axs[0].set_ylabel(r'Continuation alignment, $\Delta_i$')
    axs[1].set_xlabel('Residual context increment (nats)',labelpad=7)
    axs[1].set_ylabel('Residual continuation alignment')
    fig.text(.10,.065,'595 items per panel. Both axes in b are linearly adjusted for measured word-only preference.',fontsize=8)
    save(fig,'fig4_conditional_association',list(axs))
    csv(pd.DataFrame({'sentence_id':df.sentence_id,'source_sid':df.source_sid,
        'C_i_nats':c,'S_word_i_nats':word,'Delta_i':y,
        'C_residual_nats':cr,'Delta_residual':yr}), 'fig4_items_and_residuals.csv')
    csv(pd.DataFrame({'C_i_nats':grid_a,'fitted_Delta':raw_fit,'ci_lower':raw_fit-TC*raw_se,'ci_upper':raw_fit+TC*raw_se}),'fig4a_descriptive_fit.csv')
    csv(pd.DataFrame({'C_residual_nats':grid_b,'fitted_Delta_residual':cond_fit,'ci_lower':cond_fit-TC*cond_se,'ci_upper':cond_fit+TC*cond_se}),'fig4b_conditional_fit.csv')
    csv(pd.DataFrame([
        {'panel':'a','statistic':'Pearson correlation','r':raw_r,'n_items':len(df),'n_source_sentences':len(np.unique(groups)),
         'adjustment':'none'},
        {'panel':'b','statistic':'Partial Pearson correlation','r':partial_r,'n_items':len(df),'n_source_sentences':len(np.unique(groups)),
         'adjustment':'Both variables residualized against an intercept and measured word-only preference'}
    ]),'fig4_correlation_annotations.csv')
    numerical['Figure4']={'raw_c_delta_pearson_r':raw_r,
        'partial_c_delta_pearson_r_adjusted_for_word':partial_r,
        'partial_r_formula_verification_error':float(abs(partial_r-partial_r_formula)),
        'c_word_pearson_r':c_word_r,
        'raw_descriptive_intercept':float(raw_b[0]), 'raw_descriptive_slope':float(raw_b[1]),
        'partial_slope':residual_slope,'M1_frozen_slope':float(b1[1]),
        'residual_C_projection_intercept_slope':c_projection.tolist(),
        'residual_Delta_projection_intercept_slope':y_projection.tolist(),
        'band':'M1 frozen CR1 slope uncertainty; nuisance projections held fixed; anchored at residual origin'}

    materials=pd.read_csv(ROOT/'data'/'processed'/'analysis_items.csv')
    table1=[]
    for genre,label in [('ACPROSE','Academic prose'),('FICTION','Fiction'),('NEWS','News')]:
        g=materials.loc[materials.genre.eq(genre)]
        table1.append([label,g.source_sid.nunique(),g.sentence_id.nunique(),len(g)])
    table1.append(['Total',materials.source_sid.nunique(),materials.sentence_id.nunique(),len(materials)])
    assert table1[-1][1:]==[553,595,880]
    t1=pd.DataFrame(table1,columns=['Genre','Source sentences','Target items','M/A/I triples'])
    csv(t1,'table1_material_counts.csv')
    table_canvas('table1_materials','Table 1 | Retained materials by genre',list(t1.columns),table1,[.34,.22,.22,.22],
        'Source sentence: source_sid. Target item: sentence_id, with a marked target.\nEach M/A/I triple contains one metaphor and its two candidate substitutions.',75)

    table2=[]
    for model,spec in [('M0',m0),('M1',m1)]:
        names={'beta_0':'Intercept','beta_total':'Contextual preference','gamma_0':'Intercept',
               'gamma_C':'Context increment','gamma_W':'Word-only preference'}
        for key,co in spec['coefficients'].items():
            ci=co['confidence_interval_95']; test=co['test']
            table2.append({'Model':model,'Term':names[key],'coefficient_key':key,
                'Estimate':co['estimate'],'CR1 SE':co['cr1_standard_error'],
                'CI lower':ci['lower'],'CI upper':ci['upper'],'t':test['t_statistic'],
                'p two-sided':test['p_value'],'R_squared':spec['r_squared']})
    t2=pd.DataFrame(table2)
    csv(t2,'table2_regression_full_precision.csv')
    displayed=[]
    for row in table2:
        displayed.append([f"{row['Model']}: {row['Term']}", f"{row['Estimate']:.6f}",f"{row['CR1 SE']:.6f}",
            f"[{row['CI lower']:.6f}, {row['CI upper']:.6f}]",f"{row['t']:.3f}",f"{row['p two-sided']:.2e}"])
    table_canvas('table2_rq2_models','Table 2 | RQ2 regression results',
        ['Model / predictor','Estimate','CR1 SE','95% CI','t','p'],displayed,
        [.30,.12,.12,.25,.075,.135],
        'Both models: 595 target items; 553 source-sentence clusters; t reference df = 552.\n'
        'Outcome: continuation alignment. OLS estimates; CR1 clustered uncertainty; two-sided p values.\n'
        'M0: R² = 0.0402. M1: R² = 0.0462. Preference predictors are measured in nats.',100)

    md='# Main Tables\n\n## Table 1 | Retained materials by genre and level\n\n| Genre | Source sentences | Target items | Triples |\n|---|---:|---:|---:|\n'
    labels={'Academic prose':'Academic prose','Fiction':'Fiction','News':'News','Total':'Total'}
    for row in table1:
        md += '| '+ ' | '.join([labels[row[0]]]+[str(n) for n in row[1:]])+' |\n'
    md+='\nNote: source sentences are identified by source_sid; target items by sentence_id, with a designated target word. Each triple contains the M, A and I conditions. The three levels are counted separately and are not additive. Analyses give equal weight to 595 target items and cluster uncertainty by 553 source sentences.\n\n'
    md+='## Table 2 | Regression results for the two RQ2 models\n\n| Model / predictor | Estimate | CR1 SE | 95% CI | t | Two-sided p |\n|---|---:|---:|---|---:|---:|\n'
    for row in displayed:
        md+='| '+' | '.join(row)+' |\n'
    md+='\nNote: both models predict continuation alignment, Δᵢ, from 595 target items nested within 553 source sentences. Coefficients use ordinary least squares (OLS). CR1 cluster-robust standard errors allow within-source dependence. Intervals are 95% confidence intervals based on a Student t reference distribution with 552 degrees of freedom. All p values are two-sided and unadjusted for multiple comparisons. M0 includes contextual preference; M1 includes the context increment and word-only preference. All preference scores are natural-log response-probability contrasts in nats. R-squared is 0.0402 for M0 and 0.0462 for M1. Displayed values are rounded; Source Data retain full precision and covariance matrices.\n'
    (OUT/'tables'/'tables.md').write_text(md,encoding='utf-8')
    write_json(OUT/'source_data'/'rq2_frozen_results.json',js(RUN/'rq2'/'rq2_results.json'))
    write_json(OUT/'source_data'/'rq1_frozen_results.json',js(RUN/'primary'/'primary_analysis_results.json'))
    write_json(validation_path(OUT, 'main_numerical_validation.json'),{'passed':True,'N':595,'G':553,'exclusions':0,'checks':numerical})
    inputs=[data_path,RUN/'rq2'/'rq2_results.json',RUN/'primary'/'primary_analysis_results.json',
            RUN/'distances'/'qwen3p5_9b_formal_sentence_distances.csv',ROOT/'data'/'processed'/'analysis_items.csv']
    write_json(validation_path(OUT, 'main_provenance.json'),{'run':'20260908T224807Z','model':'Qwen/Qwen3.5-9B',
        'inputs':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,
        'display_formats':['PDF','SVG (editable text)','PNG 300 dpi','TIFF 600 dpi']})
    print('Main figures 2–4 and tables 1–2 exported; frozen coefficients and CR1 covariance reproduced.')


if __name__=='__main__':
    main()
