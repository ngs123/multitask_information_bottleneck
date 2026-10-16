"""Readable and LaTeX-ready exports of the robustness summary tables."""
import csv
import html
from pathlib import Path


def display(value):
    if value is None or str(value) in ('','None','null','nan'): return '—'
    try:return f'{float(value):.6g}'
    except (TypeError,ValueError):return str(value)


def latex(value):
    mapping={'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#',
             '{':r'\{','}':r'\}','$':r'\$','~':r'\textasciitilde{}','^':r'\textasciicircum{}','—':'--'}
    return ''.join(mapping.get(ch,ch) for ch in display(value))


def write_tables(folder, announce=False):
    folder=Path(folder);all_md=[];all_html=[];all_tex=[]
    for stem,title in [('fitted_quantities','Fitted quantities and crossover'),('rank_instability','Pretext rank instability')]:
        path=folder/f'{stem}.csv'
        if not path.exists():continue
        with path.open() as f:records=list(csv.DictReader(f))
        if not records:continue
        if stem=='fitted_quantities':
            headers=['Quantity','Value'];rows=[[r['quantity'],r['value']] for r in records]
        else:
            headers=['Panel','Low rate','High rate','Kendall tau-b','Reversed / total pairs','Comparable pairs','Max rank change']
            rows=[[r['panel'],r['low_rate'],r['high_rate'],r['kendall_tau_b'],
                   f"{r['reversed_pairs']} / {r['total_pairs']}",r['comparable_untied_pairs'],r['max_rank_change']] for r in records]
        rows=[[display(v) for v in row] for row in rows]
        md='## '+title+'\n\n'+'| '+' | '.join(headers)+' |\n'+'| '+' | '.join(['---']*len(headers))+' |\n'
        md+=''.join('| '+' | '.join(v.replace('|',r'\|').replace('\n',' ') for v in row)+' |\n' for row in rows)
        h='<h2>'+html.escape(title)+'</h2><table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>'
        h+=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
        tex='\\begin{tabular}{'+'l'*len(headers)+'}\n\\hline\n'+' & '.join(latex(v) for v in headers)+r' \\'+'\n\\hline\n'
        tex+=''.join(' & '.join(latex(v) for v in row)+r' \\'+'\n' for row in rows)+'\\hline\n\\end{tabular}\n'
        (folder/f'{stem}.md').write_text(md);(folder/f'{stem}.tex').write_text(tex)
        all_md.append(md);all_html.append(h);all_tex.append('% '+title+'\n'+tex)
    note=('Crossover intervals are conditional on observing an interior crossing; report the crossing fraction. '
          'The isotropic rate threshold is a sufficient bound, not a predicted crossing. '
          'Unavailable values are shown as an em dash. Check moment_bootstrap/status.json for completeness.')
    (folder/'tables.md').write_text('\n\n'.join(all_md)+'\n\n'+note+'\n')
    (folder/'tables.tex').write_text('\n\n'.join(all_tex))
    (folder/'tables.html').write_text('<!doctype html><meta charset="utf-8"><title>Rate selection tables</title>'
        '<style>body{font:15px sans-serif;margin:32px}table{border-collapse:collapse;margin-bottom:28px}'
        'th,td{border:1px solid #ccc;padding:8px 12px;text-align:left}th{background:#eee}</style>'
        +''.join(all_html)+'<p>'+html.escape(note)+'</p>')
    if announce:
        print('\n'+'\n\n'.join(all_md),flush=True)
        print(f'Readable tables: {(folder/"tables.html").resolve()}\nLaTeX tables: {(folder/"tables.tex").resolve()}',flush=True)
