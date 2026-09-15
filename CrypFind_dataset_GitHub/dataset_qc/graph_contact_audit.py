#!/usr/bin/env python3
"""Streaming, restartable graph schema/QC and Δd audit; original .pt files are never changed."""
import argparse,csv,json,math,os,traceback
from pathlib import Path
from collections import Counter
import torch
import numpy as np
ROOT=Path(os.environ.get('CRYPFIND_DATA_ROOT', '.')).resolve(); OUT=Path(os.environ.get('CRYPFIND_QC_OUT', ROOT/'qc_output')).resolve()
BINS=np.linspace(-100,200,60001,dtype=np.float64)
def sample_id(p): return p.stem
def worker(p):
 try:
  d=torch.load(p,map_location='cpu',weights_only=False); req=['apo_pos','holo_pos','edge_index','edge_label','sequence']
  miss=[x for x in req if not hasattr(d,x)]
  n=int(d.num_nodes); e=d.edge_index
  if miss or d.apo_pos.ndim!=2 or d.apo_pos.shape[1]!=3 or d.holo_pos.shape!=d.apo_pos.shape or e.ndim!=2 or e.shape[0]!=2 or d.edge_label.numel()!=e.shape[1]: return ('bad_schema',str(p),n,0,0,0,0,0,0,0,0,0,0,[],None)
  if n==0:return ('node0',str(p),n,0,0,0,0,0,0,0,0,0,0,[],None)
  if e.shape[1]==0:return ('edge0',str(p),n,0,0,0,0,0,0,0,0,0,0,[],None)
  a=e[0].numpy(); b=e[1].numpy(); keys=np.minimum(a,b).astype(np.int64)*max(n,1)+np.maximum(a,b); ix=np.unique(keys,return_index=True)[1]
  a=a[ix]; b=b[ix]; da=torch.linalg.vector_norm(d.apo_pos[a]-d.apo_pos[b],dim=1).numpy(); dh=torch.linalg.vector_norm(d.holo_pos[a]-d.holo_pos[b],dim=1).numpy(); dd=dh-da; rec=(dd>3.0); stored=d.edge_label.numpy()[ix].astype(bool)
  hist=np.histogram(dd,bins=BINS)[0]; return ('ok',str(p),n,int(e.shape[1]),len(dd),int(rec.sum()),int(stored.sum()),int((rec==stored).sum()),len(dd),float(dd.mean()),float(np.median(dd)),float(dd.min()),float(dd.max()),hist,None)
 except Exception as ex:return ('unreadable',str(p),0,0,0,0,0,0,0,0,0,0,0,[],f'{type(ex).__name__}: {ex}')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=0);args=ap.parse_args()
 files=sorted((ROOT/'processed_pairs_backbone').glob('*.pt')); end=args.end or len(files); files=files[args.start:end]
 part=OUT/'intermediate'/f'delta_part_{args.start}_{end}.csv'; err=OUT/'intermediate'/f'delta_errors_{args.start}_{end}.csv'; OUT.joinpath('intermediate').mkdir(parents=True,exist_ok=True)
 hist=np.zeros(len(BINS)-1,dtype=np.int64); counts=Counter(); f=part.open('w',newline='');w=csv.writer(f);w.writerow(['sample_id','n_nodes','n_directed_edges','n_unique_apo_contacts','n_disrupted_contacts','stored_positive_unique','label_agree_unique','disrupted_fraction','median_delta_d','mean_delta_d','min_delta_d','max_delta_d','status'])
 ef=err.open('w',newline='');ew=csv.writer(ef);ew.writerow(['path','status','error'])
 for j,p in enumerate(files,1):
  x=worker(p); status,path,n,nd,nu,np_,sp,agree,comp,mean,med,mn,mx,h,e=x; counts[status]+=1
  if status=='ok':hist+=h; counts['nodes']+=n;counts['directed']+=nd;counts['unique']+=nu;counts['positive']+=np_;counts['stored_pos']+=sp;counts['agree']+=agree; w.writerow([sample_id(p),n,nd,nu,np_,sp,agree,np_/nu,med,mean,mn,mx,status])
  else: ew.writerow([path,status,e]);w.writerow([sample_id(p),n,nd,nu,np_,sp,agree,'','','','','',status])
  if j%1000==0: print(f'{args.start+j}/{end}',flush=True)
 f.close();ef.close();np.savez_compressed(OUT/'intermediate'/f'delta_hist_{args.start}_{end}.npz',bins=BINS,hist=hist)
 json.dump(dict(counts),(OUT/'intermediate'/f'delta_counts_{args.start}_{end}.json').open('w'))
if __name__=='__main__':main()
