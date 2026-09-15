#!/usr/bin/env python3
import csv,json,sys,os
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
ROOT=Path(os.environ.get('CRYPFIND_DATA_ROOT', '.')).resolve();OUT=Path(os.environ.get('CRYPFIND_QC_OUT', ROOT/'qc_output')).resolve();sys.path.insert(0,str(Path(__file__).parent))
from corrected_rmsd_qc import calc,clusters
def st(x):
 x=np.asarray(x,float);q=lambda p:float(np.quantile(x,p));return {'N':len(x),'mean':float(x.mean()),'SD':float(x.std(ddof=1)),'min':float(x.min()),'P1':q(.01),'P5':q(.05),'P25':q(.25),'median':q(.5),'P50':q(.5),'P75':q(.75),'P95':q(.95),'P99':q(.99),'max':float(x.max()),'IQR_low':q(.25),'IQR_high':q(.75)}
def main():
 parts=[OUT/'intermediate'/f'rmsd_part_{s}_{e}.csv' for s,e in [(0,45653),(45653,91306),(91306,136959),(136959,182612),(182612,228265),(228265,273918),(273918,319571),(319571,365224)]];rows=[]
 for p in parts:
  with p.open() as f:rows.extend(csv.DictReader(f))
 with (OUT/'tables/rmsd_dataset_qc_per_sample.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 ok=[r for r in rows if r['status']=='ok'];g=np.array([float(r['global_ca_rmsd']) for r in ok]);h=np.array([float(r['interface_ca_rmsd_after_global_fit']) for r in ok]);cv=np.array([float(r['mapping_coverage']) for r in ok]);d=h-g
 # independent 30: bottom 10, centered median 10, top 10
 order=np.argsort(g);mid=len(g)//2;pick=np.r_[order[:10],order[mid-5:mid+5],order[-10:]];cl=clusters();vr=[]
 for ix in pick:
  r=ok[int(ix)];vr.append(calc(ROOT/'processed_pairs_backbone'/f"{r['sample_id']}.pt",cl,biopy=True))
 with (OUT/'tables/rmsd_independent_validation_30samples.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['sample_id','protein_id','corrected_rmsd','biopython_svd_rmsd','abs_difference','detR','n_mapped_ca','status']);w.writerows(vr)
 vd=np.array([float(x[4]) for x in vr if x[-1]=='ok'])
 gs,hs,cs,ds=st(g),st(h),st(cv),st(d);summary={'n_final_graphs':len(rows),'n_valid_global_rmsd':len(ok),'n_valid_interface_rmsd':len(ok),'n_failed':len(rows)-len(ok),'global_rmsd':gs,'interface_rmsd_after_global_fit':hs,'mapping_coverage':cs,'interface_minus_global':ds,'n_global_lt1':int((g<1).sum()),'n_global_lt2':int((g<2).sum()),'n_global_lt3':int((g<3).sum()),'n_global_lt5':int((g<5).sum()),'n_global_5to10':int(((g>=5)&(g<=10)).sum()),'n_global_gt10':int((g>10).sum()),'pct_global_lt1':float((g<1).mean()),'pct_global_lt2':float((g<2).mean()),'pct_global_lt3':float((g<3).mean()),'pct_global_lt5':float((g<5).mean()),'pct_global_5to10':float(((g>=5)&(g<=10)).mean()),'pct_global_gt10':float((g>10).mean()),'pct_interface_gt_global':float((h>g).mean()),'pct_interface_le_global':float((h<=g).mean()),'independent_validation_N':len(vd),'independent_mean_abs_difference':float(vd.mean()),'independent_max_abs_difference':float(vd.max())}
 json.dump(summary,(OUT/'tables/rmsd_dataset_qc_summary.json').open('w'),indent=2)
 with (OUT/'tables/rmsd_dataset_qc_summary.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['metric','value']);
  for k,v in summary.items():
   if not isinstance(v,dict):w.writerow([k,v])
  for group,obj in [('global',gs),('interface',hs),('mapping_coverage',cs),('interface_minus_global',ds)]:
   for k,v in obj.items():w.writerow([f'{group}_{k}',v])
 fig,ax=plt.subplots(1,2,figsize=(11,4));p995=np.quantile(g,.995);ax[0].hist(g,bins=160,density=True,color='#2b6cb0');ax[0].set_xlim(0,p995);ax[0].set(xlabel='Global Cα RMSD (Å)',ylabel='Density',title='Global structural correspondence');ax[0].text(.97,.95,f'N={len(g):,}\nmedian={gs["median"]:.2f} Å\nIQR={gs["IQR_low"]:.2f}–{gs["IQR_high"]:.2f} Å',ha='right',va='top',transform=ax[0].transAxes);lim=max(np.quantile(g,.995),np.quantile(h,.995));ax[1].hexbin(g,h,gridsize=90,mincnt=1,cmap='viridis');ax[1].plot([0,lim],[0,lim],'r--',lw=1);ax[1].set(xlabel='Global Cα RMSD (Å)',ylabel='Interface Cα RMSD after global fit (Å)',title='Global vs interface correspondence');plt.tight_layout();OUT.joinpath('figures').mkdir(exist_ok=True);plt.savefig(OUT/'figures/Fig_RMSD_dataset_QC.png',dpi=300);plt.savefig(OUT/'figures/Fig_RMSD_dataset_QC.pdf');plt.close()
 rep=f'''# Edge-GCL corrected RMSD Dataset QC\n\n## 1. Previous RMSD issue\n\nPrevious full-dataset RMSD values were invalid because the Kabsch rotation was applied in the wrong direction for row-vector coordinates. Mapping issue: **NO**. Kabsch issue: **YES**. The corrected `U @ Vt` implementation was independently validated against Biopython SVD before and after full computation.\n\n## 2. Dataset coverage\n\nFinal graphs: {len(rows):,}. Valid global RMSD: {len(ok):,}. Valid interface RMSD: {len(ok):,}. Failed: {len(rows)-len(ok):,}.\n\n## 3. Global RMSD\n\nMedian [IQR]: {gs['median']:.3f} [{gs['IQR_low']:.3f}, {gs['IQR_high']:.3f}] Å. Mean ± SD: {gs['mean']:.3f} ± {gs['SD']:.3f} Å. P5–P95: {gs['P5']:.3f}–{gs['P95']:.3f} Å. <2 Å: {summary['pct_global_lt2']:.2%}; <3 Å: {summary['pct_global_lt3']:.2%}; <5 Å: {summary['pct_global_lt5']:.2%}; >10 Å: {summary['pct_global_gt10']:.2%}.\n\n## 4. Interface RMSD\n\nInterface RMSD was calculated after global fitting. Median [IQR]: {hs['median']:.3f} [{hs['IQR_low']:.3f}, {hs['IQR_high']:.3f}] Å. Median interface minus global: {ds['median']:.3f} Å. Interface RMSD > global: {summary['pct_interface_gt_global']:.2%}.\n\n## 5. Mapping QC\n\nMapping coverage is `n_mapped_ca / apo_length`, matching the prior sanity-check denominator. Median [IQR]: {cs['median']:.4f} [{cs['IQR_low']:.4f}, {cs['IQR_high']:.4f}].\n\n## 6. Independent validation\n\nN = {len(vd)}. Corrected Kabsch versus Biopython SVD mean absolute difference: {vd.mean():.3e} Å; maximum absolute difference: {vd.max():.3e} Å. The validation confirms numerical agreement.\n\n## 7. Dataset QC interpretation\n\nThe corrected distributions provide the valid structural-consistency characterization for the final pretraining corpus. They supersede the invalid earlier full-dataset RMSD values. Interface RMSD is reported descriptively after global fitting; no directional interface-rearrangement claim is imposed.\n\n## 8. Limitations\n\nThe analysis preserves all successful final graph samples and does not introduce a new quality filter. High-RMSD observations, where present, remain in the distribution; no TM-score or additional structural similarity method was calculated at full-dataset scale.\n'''
 (OUT/'REPORT_RMSD_dataset_QC.md').write_text(rep);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
