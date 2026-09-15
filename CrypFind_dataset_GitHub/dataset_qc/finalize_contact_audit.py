#!/usr/bin/env python3
import csv,json,glob,os
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('CRYPFIND_DATA_ROOT', '.')).resolve();O=Path(os.environ.get('CRYPFIND_QC_OUT', ROOT/'qc_output')).resolve(); I=O/'intermediate';T=O/'tables';T.mkdir(parents=True,exist_ok=True)
bins=None;hist=None;c={}
for p in sorted(I.glob('delta_counts_*.json')):
 for k,v in json.load(p.open()).items():c[k]=c.get(k,0)+v
for p in sorted(I.glob('delta_hist_*.npz')):
 z=np.load(p); bins=z['bins'];hist=z['hist'].astype('int64') if hist is None else hist+z['hist']
with (T/'delta_d_per_graph_summary.csv').open('w',newline='') as out:
 w=csv.writer(out); first=True
 for p in sorted(I.glob('delta_part_*.csv')):
  for r in csv.reader(p.open()):
   if first or r[0]!='sample_id':w.writerow(r)
   first=False
np.savetxt(T/'delta_d_histogram.csv',np.c_[bins[:-1],bins[1:],hist],delimiter=',',header='bin_left,bin_right,count',comments='')
cum=np.cumsum(hist);n=int(cum[-1]);
def q(x):return float(bins[min(np.searchsorted(cum,n*x),len(bins)-2)])
summary={'valid_graphs':c.get('ok',0),'unreadable':c.get('unreadable',0),'bad_schema':c.get('bad_schema',0),'node0':c.get('node0',0),'edge0':c.get('edge0',0),'directed_edges':c.get('directed',0),'unique_undirected_contacts':c.get('unique',0),'positive_unique_contacts':c.get('positive',0),'positive_fraction':c.get('positive',0)/max(1,c.get('unique',0)),'stored_recomputed_label_agreement':c.get('agree',0)/max(1,c.get('unique',0)),'delta_d_histogram_bin_width':float(bins[1]-bins[0]),'delta_d_quantiles_histogram_approx':{str(k):q(k) for k in [.01,.05,.25,.5,.75,.95,.99]}}
json.dump(summary,(T/'delta_d_summary.json').open('w'),indent=2);print(json.dumps(summary,indent=2))
