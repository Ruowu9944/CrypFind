#!/usr/bin/env python3
"""Publication layout for Figure 2; reads existing characterization summaries only."""
import csv,json,logging,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(os.environ.get('CRYPFIND_DATA_ROOT', '.')).resolve(); SRC=Path(os.environ.get('CRYPFIND_QC_TABLES', ROOT/'qc_output/tables')).resolve(); OUT=Path(os.environ.get('CRYPFIND_FIGURE_OUT', ROOT/'figure_output')).resolve()
BLUE='#6F8FA8'; BLUE_DARK='#4F718C'; ORANGE='#C87522'; PALE='#F5E9D8'; GREY='#6E6E6E'; DARK='#151515'
def rows(p):
 with open(p) as f:return list(csv.DictReader(f))
def style():
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.55,'xtick.major.width':.55,'ytick.major.width':.55,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
def save(fig,name):
 for e,kw in [('.pdf',{}),('.svg',{}),('.png',{'dpi':600})]:fig.savefig(OUT/'figures'/f'{name}{e}',bbox_inches='tight',**kw)
def quant(c,count,p):return c[min(np.searchsorted(np.cumsum(count),p*count.sum()),len(c)-1)]
def load():
 s={r['stage']:int(float(r['count'])) for r in rows(SRC/'dataset_stage_counts.csv')}; d=json.load(open(SRC/'delta_d_summary.json'));e={r['reason']:int(float(r['count'])) for r in rows(SRC/'preprocessing_exclusion_reasons.csv')};h=np.loadtxt(SRC/'delta_d_histogram.csv',delimiter=',',skiprows=1);f=[]
 for r in rows(SRC/'delta_d_per_graph_summary.csv'):
  if r['status']=='ok':f.append(float(r['disrupted_fraction']))
 return s,d,e,h,np.asarray(f)
def card(ax,x,num,label):
 ax.add_patch(FancyBboxPatch((x,.790),.195,.115,boxstyle='round,pad=.008,rounding_size=.015',fc='#F3F5F6',ec='#333',lw=.7,transform=ax.transAxes));ax.text(x+.0975,.868,num,ha='center',va='center',fontsize=11.5,fontweight='medium',transform=ax.transAxes);ax.text(x+.0975,.816,label,ha='center',va='center',fontsize=6.2,transform=ax.transAxes)
def panel_a(ax,s,d,e):
 ax.axis('off');ax.text(.00,.965,'A',fontsize=11,fontweight='bold',transform=ax.transAxes);ax.text(.032,.965,'Dataset construction yield and quality control',fontsize=9.2,fontweight='bold',transform=ax.transAxes)
 card(ax,.035,f'{s["high_quality_homodimers"]:,}','High-confidence\nhomodimer candidates');card(ax,.397,f'{s["UniRef50_representatives"]:,}','UniRef50\nrepresentatives');card(ax,.758,f'{s["existing_pt_graph_files"]:,}','Valid paired graphs')
 for x1,x2,t in [(.238,.385,'29.69% retained'),(.600,.746,'80.31% of Stage B')]:ax.add_patch(FancyArrowPatch((x1,.852),(x2,.852),arrowstyle='-|>',mutation_scale=10,lw=.8,color=DARK,transform=ax.transAxes));ax.text((x1+x2)/2,.860,t,ha='center',va='bottom',fontsize=6.5,transform=ax.transAxes)
 ax.text(.673,.803,'23.84% overall',fontsize=6.2,transform=ax.transAxes)
 ax.text(.16,.705,'exclusion summary',ha='center',fontsize=8,fontweight='bold',transform=ax.transAxes);ax.text(.16,.655,'Exclusions after UniRef50 selection, n = 89,566',ha='center',fontsize=6.8,transform=ax.transAxes)
 sub=ax.inset_axes([.04,.335,.37,.27]);labs=['Missing holo-like structure','Missing apo-like structure','Too few valid nodes'];vals=[e['missing_holo'],e['missing_apo'],e['too_few_nodes']];yy=np.arange(3);sub.barh(yy,vals,color=[BLUE,BLUE,'#C9D2D8'],height=.55);sub.set_yticks(yy,labs);sub.invert_yaxis();sub.set_xlabel('Excluded candidates',labelpad=1);sub.ticklabel_format(axis='x',style='sci',scilimits=(3,3));sub.tick_params(axis='y',labelsize=6,length=0);sub.spines[['top','right','left']].set_visible(False)
 for y,v in zip(yy,vals):sub.text(v+max(vals)*.015,y,f'{v:,}',va='center',fontsize=6)
 ax.text(.74,.705,'dataset summary',ha='center',fontsize=8,fontweight='bold',transform=ax.transAxes)
 facts=[('365,221','paired graphs',DARK),('48.21 M','unique apo contacts',DARK),('1.03 M','disrupted contacts',ORANGE)]
 for x,(n,l,c) in zip([.51,.68,.85],facts):ax.text(x,.56,n,ha='center',fontsize=13,fontweight='medium',color=c,transform=ax.transAxes);ax.text(x,.495,l,ha='center',fontsize=6.8,color=c,transform=ax.transAxes)
 ax.text(.76,.42,'96.43 M directed graph edges',ha='center',fontsize=6.2,color=GREY,transform=ax.transAxes)
def panel_b(ax,h,d):
 ax.text(-.10,1.15,'B',transform=ax.transAxes,fontsize=11,fontweight='bold');ax.text(-.02,1.15,'Distribution of residue-contact distance changes',transform=ax.transAxes,fontsize=8.5,fontweight='bold')
 c=(h[:,0]+h[:,1])/2;cnt=h[:,2];lo=quant(c,cnt,.01);hi=quant(c,cnt,.98);m=(c>=lo)&(c<=hi); # coarsen existing 0.005-A histogram into display bins
 edges=np.linspace(lo,hi,61);b,_=np.histogram(c[m],bins=edges,weights=cnt[m]);width=np.diff(edges);density=b/(cnt.sum()*width);ax.bar(edges[:-1],density,width=width,align='edge',color=BLUE,edgecolor=BLUE_DARK,lw=.25);ax.axvspan(3,hi,color=PALE,zorder=0);ax.axvline(0,color=GREY,ls='--',lw=.7);ax.axvline(3,color=ORANGE,ls='--',lw=.9);ax.text(3.05,ax.get_ylim()[1]*.66,'Disruption\nthreshold',color=ORANGE,fontsize=6.5);ax.text((3+hi)/2,ax.get_ylim()[1]*.20,'Disrupted\ncontacts',ha='center',color=DARK,fontsize=6.5)
 ax.set(xlabel='Contact distance change, Δd (Å)\n(Δd = d$_{interface-associated}$ − d$_{apo-like}$)',ylabel='Density',xlim=(lo,hi));ax.spines[['top','right']].set_visible(False);ax.text(.04,.94,f'N = {int(d["unique_undirected_contacts"]):,}\nunique contacts\n\nMedian Δd = {d["delta_d_quantiles_histogram_approx"]["0.5"]:.3f} Å\nΔd >3 Å = {d["positive_fraction"]:.2%}',transform=ax.transAxes,va='top',fontsize=6.5)
 ins=ax.inset_axes([.69,.69,.26,.25]);ins.plot(c,cnt/cnt.sum(),color=BLUE_DARK,lw=.55);ins.axvline(3,color=ORANGE,ls='--',lw=.55);ins.set_title('Full-range\ndistribution',fontsize=5,pad=1);ins.tick_params(labelsize=3.8,length=1);ins.set_yticks([]);ins.set_xlim(c.min(),c.max())
 return lo,hi
def panel_c(ax,d,f):
 ax.axis('off');ax.text(.00,.965,'C',fontsize=11,fontweight='bold',transform=ax.transAxes);ax.text(.055,.965,'Sparsity of contact-level disruption events',fontsize=8.5,fontweight='bold',transform=ax.transAxes)
 n=int(d['unique_undirected_contacts']);p=int(d['positive_unique_contacts']);non=n-p;ax.text(.06,.815,'C1: Contact-level composition',fontsize=7.7,fontweight='bold',transform=ax.transAxes)
 b=ax.inset_axes([.08,.665,.86,.105]);b.barh([0],[non],color=BLUE,height=.65);b.barh([0],[p],left=[non],color=ORANGE,height=.65);b.set_xlim(0,n);b.axis('off');b.text(non*.48,0,f'Non-disrupted contacts\n{non:,}, {(non/n):.2%}',color='white',ha='center',va='center',fontsize=5.8);ax.text(.94,.61,f'Disrupted contacts\n{p:,}, {(p/n):.2%}',color=ORANGE,ha='right',va='top',fontsize=6.2,transform=ax.transAxes,zorder=10)
 ax.text(.06,.48,'C2: Per-graph disruption fraction',fontsize=7.7,fontweight='bold',transform=ax.transAxes);q95=float(np.quantile(f,.95));upper=float(np.quantile(f,.995));edges=np.r_[0,1e-12,np.linspace(upper/65,upper,66)];co,_=np.histogram(f,bins=edges);g=ax.inset_axes([.08,.10,.86,.29]);g.bar(edges[:-1],co/len(f),width=np.diff(edges),align='edge',color=BLUE,edgecolor=BLUE_DARK,lw=.2);g.set(xlabel='Fraction of disrupted contacts per graph',ylabel='Fraction of graphs',xlim=(0,upper));g.spines[['top','right']].set_visible(False);g.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0));ge=int((f>0).sum());ax.text(.94,.46,f'Median = 0\n95th percentile = {q95:.2%}\nGraphs with ≥1\ndisrupted contact:\nN = {ge:,} ({ge/len(f):.2%})',ha='right',va='top',fontsize=6.2,transform=ax.transAxes)
 return non,ge,q95
def main():
 OUT.joinpath('figures').mkdir(parents=True,exist_ok=True);OUT.joinpath('tables').mkdir(exist_ok=True);OUT.joinpath('logs').mkdir(exist_ok=True);logging.basicConfig(filename=OUT/'logs/make_figure2.log',level=logging.INFO);style();s,d,e,h,f=load();fig=plt.figure(figsize=(7.09,5.35));a=fig.add_axes([.04,.51,.92,.45]);b=fig.add_axes([.07,.11,.40,.27]);c=fig.add_axes([.54,.11,.42,.27]);panel_a(a,s,d,e);rng=panel_b(b,h,d);non,ge,q95=panel_c(c,d,f);save(fig,'Figure2_dataset_characterization');
 # individual panels retained for later editing
 for name,fun,args,size in [('Figure2A_dataset_yield',panel_a,(s,d,e),(7.09,2.8)),('Figure2B_delta_d',panel_b,(h,d),(4.2,3.0)),('Figure2C_disruption_sparsity',panel_c,(d,f),(4.2,3.0))]:
  ff,aa=plt.subplots(figsize=size);fun(aa,*args);save(ff,name)
 cap=f'''Figure 2. Quantitative characterization of the Edge-GCL pretraining dataset. (A) Dataset construction yield and quality control. High-confidence homodimer candidates were reduced to UniRef50 representatives, of which {s['existing_pt_graph_files']:,} were successfully converted into valid paired graphs. Most exclusions resulted from unavailable holo-like or apo-like structures. (B) Distribution of residue-contact distance changes (Δd) across unique apo-state contacts. Most contacts showed little distance change, whereas {d['positive_fraction']:.2%} exceeded the 3-Å disruption threshold. (C) Contact- and graph-level sparsity of disruption events, showing the proportion of disrupted versus non-disrupted contacts and the distribution of disrupted-contact fractions across individual graphs.\n''';(OUT/'figures/Figure2_caption.txt').write_text(cap)
 src={'final_graphs':s['existing_pt_graph_files'],'unique_contacts':int(d['unique_undirected_contacts']),'disrupted_contacts':int(d['positive_unique_contacts']),'non_disrupted_contacts':non,'graphs_with_ge1_disruption':ge,'pct_graphs_with_ge1_disruption':ge/len(f),'p95_graph_disruption_fraction':q95,'delta_display_range':rng,'inset_used':True};json.dump(src,open(OUT/'tables/figure2_source_statistics.json','w'),indent=2);print(json.dumps(src,indent=2))
if __name__=='__main__':main()
