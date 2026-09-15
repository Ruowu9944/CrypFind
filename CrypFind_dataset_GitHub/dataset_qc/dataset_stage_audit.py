#!/usr/bin/env python3
"""Read-only Stage A/B/C audit for Edge-GCL characterization."""
import csv, json, os, sys
from collections import Counter
from pathlib import Path

ROOT=Path(os.environ.get('CRYPFIND_DATA_ROOT', '.')).resolve()
OUT=Path(os.environ.get('CRYPFIND_QC_OUT', ROOT/'qc_output')).resolve()

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def q(vals, p):
    vals=sorted(vals); n=len(vals)
    if not n:return None
    x=(n-1)*p; a=int(x); b=min(a+1,n-1); return vals[a]+(vals[b]-vals[a])*(x-a)
def stats(vals):
    vals=[float(v) for v in vals if v not in ('',None)]
    return {'n':len(vals),'min':min(vals) if vals else None,'p1':q(vals,.01),'p5':q(vals,.05),'p25':q(vals,.25),'median':q(vals,.5),'p75':q(vals,.75),'p95':q(vals,.95),'p99':q(vals,.99),'max':max(vals) if vals else None,'mean':sum(vals)/len(vals) if vals else None}
def main():
    a=read_csv(ROOT/'AlphaFold_Data/global_high_quality_targets.csv')
    b=read_csv(ROOT/'AlphaFold_Data/refined_homo_pretrain_final.csv')
    pts=list((ROOT/'processed_pairs_backbone').glob('*.pt'))
    a_ids=[r.get('modelEntityId','') for r in a]; b_ids=[r.get('modelEntityId','') for r in b]
    summary={'stage_a':{'rows':len(a),'unique_modelEntityId':len(set(a_ids)),'duplicate_rows_by_modelEntityId':len(a_ids)-len(set(a_ids)),'ipTM':stats([r.get('ipTM') for r in a]),'pDockQ':stats([r.get('pDockQ') for r in a])},'stage_b':{'rows':len(b),'unique_modelEntityId':len(set(b_ids)),'unique_uniprot':len({r.get('uniprotAccession') for r in b}),'unique_uniref50':len({r.get('cluster_id') for r in b}),'clusters_with_gt1':sum(v>1 for v in Counter(r.get('cluster_id') for r in b).values()),'retention_vs_a':len(b)/len(a)},'stage_c':{'pt_files':len(pts),'success_vs_b':len(pts)/len(b),'overall_vs_a':len(pts)/len(a)}}
    (OUT/'tables').mkdir(parents=True,exist_ok=True)
    with (OUT/'tables/dataset_stage_counts.csv').open('w',newline='') as f:
      w=csv.writer(f);w.writerow(['stage','count','retention_rate']);w.writerows([['high_quality_homodimers',len(a),1],['UniRef50_representatives',len(b),len(b)/len(a)],['existing_pt_graph_files',len(pts),len(pts)/len(a)]])
    skips=Counter()
    p=ROOT/'build_pyg_graphs_backbone_skips.csv'
    if p.exists():
      for r in read_csv(p): skips[r.get('status') or 'unclassified preprocessing failure']+=1
    with (OUT/'tables/preprocessing_exclusion_reasons.csv').open('w',newline='') as f:
      w=csv.writer(f);w.writerow(['reason','count','percentage_of_stage_b']);w.writerows((k,v,v/len(b)) for k,v in sorted(skips.items()))
    json.dump(summary,(OUT/'tables/dataset_qc_summary.json').open('w'),indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
