#!/usr/bin/env python3
"""Streaming corrected row-vector Kabsch RMSD over final Edge-GCL graph manifest."""
import argparse,csv,sys,random
from pathlib import Path
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
ROOT=Path(__import__('os').environ.get('CRYPFIND_DATA_ROOT', '.')).resolve();OUT=Path(__import__('os').environ.get('CRYPFIND_QC_OUT', ROOT/'qc_output')).resolve();sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dataset_construction'))
from build_paired_graphs import find_apo_path,find_holo_path,load_structure,build_chain_records,sequence_from_records,build_apo_to_holo_mapping,interface_indices
def fit(a,b):
 ca=a.mean(0);cb=b.mean(0);aa=a-ca;bb=b-cb;u,s,vh=np.linalg.svd(aa.T@bb);r=u@vh
 if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vh
 return (a-ca)@r+cb,float(np.linalg.det(r))
def calc(p,clusters,biopy=False):
 sid=p.stem;u,mid=sid.rsplit('_',1);base=[sid,u,clusters.get(mid,''),'','','','','','','','']
 try:
  ap=find_apo_path(ROOT/'AlphaFold_Data/apo_monomer_structures',u);hp=find_holo_path(ROOT/'AlphaFold_Data/holo_complex_structures',mid)
  if not ap:return base+['missing_apo']
  if not hp:return base+['missing_holo']
  ac=build_chain_records(load_structure(ap));hc=build_chain_records(load_structure(hp))
  if 'A' not in hc or 'B' not in hc:return base+['bad_holo_chains']
  aid='A' if 'A' in ac else max(ac,key=lambda z:len(ac[z]));ar=ac[aid];ha=hc['A'];hb=hc['B'];mp=build_apo_to_holo_mapping(sequence_from_records(ar),sequence_from_records(ha));pairs=[(i,j) for i,j in mp.items() if ar[i].ca is not None and ha[j].ca is not None]
  if len(pairs)<3:return [sid,u,clusters.get(mid,''),len(ar),len(ha),len(pairs),len(pairs)/max(1,len(ar)),'','','','insufficient_mapped_ca']
  a=np.stack([ar[i].ca for i,j in pairs]);b=np.stack([ha[j].ca for i,j in pairs]);fa,det=fit(a,b);g=float(np.sqrt(np.mean(np.sum((fa-b)**2,axis=1))));iface=interface_indices(ha,hb,5.0);ii=[z for z,(_,j) in enumerate(pairs) if j in iface]
  if len(ii)<3:return [sid,u,clusters.get(mid,''),len(ar),len(ha),len(pairs),len(pairs)/len(ar),len(ii),g,'','','insufficient_interface_residues']
  h=float(np.sqrt(np.mean(np.sum((fa[ii]-b[ii])**2,axis=1))))
  if biopy:
   sv=SVDSuperimposer();sv.set(b,a);sv.run();bio=float(sv.get_rms());return [sid,u,g,bio,abs(g-bio),det,len(pairs),'ok']
  return [sid,u,clusters.get(mid,''),len(ar),len(ha),len(pairs),len(pairs)/len(ar),len(ii),g,h,h-g,'ok']
 except Exception as e:return base+[type(e).__name__+': '+str(e)]
def clusters():
 out={}
 with open(ROOT/'AlphaFold_Data/refined_homo_pretrain_final.csv') as f:
  for r in csv.DictReader(f):out[r['modelEntityId']]=r.get('cluster_id','')
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['validate','chunk'],required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=0);a=ap.parse_args();OUT.joinpath('tables').mkdir(parents=True,exist_ok=True);OUT.joinpath('intermediate').mkdir(parents=True,exist_ok=True);fs=sorted((ROOT/'processed_pairs_backbone').glob('*.pt'));cl=clusters()
 if a.mode=='validate':
  random.Random(20260817).shuffle(fs);r=[calc(p,cl,True) for p in fs[:10]]
  with (OUT/'tables/rmsd_preflight_biopython_10samples.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['sample_id','protein_id','corrected_rmsd','biopython_rmsd','abs_difference','detR','n_mapped_ca','status']);w.writerows(r)
  bad=[x for x in r if x[-1]!='ok' or x[4]>1e-5];print('preflight',len(r),'bad',len(bad));raise SystemExit(1 if bad else 0)
 e=a.end or len(fs);pout=OUT/'intermediate'/f'rmsd_part_{a.start}_{e}.csv'
 with pout.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['sample_id','protein_id','uniref50_id','apo_length','holo_chainA_length','n_mapped_ca','mapping_coverage','n_interface_residues','global_ca_rmsd','interface_ca_rmsd_after_global_fit','interface_minus_global','status'])
  for n,p in enumerate(fs[a.start:e],1):
   w.writerow(calc(p,cl));
   if n%1000==0:print(f'{a.start+n}/{e}',flush=True)
if __name__=='__main__':main()
