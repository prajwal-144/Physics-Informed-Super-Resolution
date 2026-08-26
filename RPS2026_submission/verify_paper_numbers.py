"""Recompute every number quoted in the paper from superres/results/ and manifest.csv.

Run in three parts; each part is standalone.
"""

# ---------------- part 1: parameter recovery ----------------
import json, csv, numpy as np
from scipy.stats import spearmanr, mannwhitneyu

R="/mnt/user-data/uploads/Grid_Based_Experiment/superres/results"
MAN="/root/.claude/uploads/ff05c4ab-0730-58da-8f88-db5c381e7d59/75ce89d4-manifest.csv"

man={}
for r in csv.DictReader(open(MAN)):
    man[r['path'].replace('/', '\\')] = r

def truth(p):
    r = man[p.replace('/', '\\')]
    f = lambda k: float(r[k])
    return dict(theta_E=f('theta_E'),
                gamma=f('host_slope'),
                e=np.hypot(f('host_e1'), f('host_e2')),
                g=np.hypot(f('gamma1_ext'), f('gamma2_ext')),
                R_sersic=f('source_R_sersic'),
                n_sersic=f('source_n_sersic'),
                beta=np.hypot(f('source_x'), f('source_y')),
                cls=r['class'])

KEYS=['theta_E','beta','R_sersic','n_sersic','gamma','e','g']

def score(fn, label):
    d=json.load(open(f"{R}/{fn}"))
    rows=d['rows']
    ok=[r for r in rows if r['path'].replace('/', '\\') in man]
    print(f"\n=== {label}  n={len(ok)} / {len(rows)}")
    T=[truth(r['path']) for r in ok]
    import collections
    print("   classes:", collections.Counter(t['cls'] for t in T))
    out={}
    for k in KEYS:
        f=np.array([r[k] for r in ok], float)
        t=np.array([x[k] for x in T], float)
        m=np.isfinite(f)&np.isfinite(t)
        rho=spearmanr(f[m], t[m]).correlation
        out[k]=rho
        print(f"   {k:9s} rho {rho:+.3f}   fit med {np.median(f[m]):.4f}  true med {np.median(t[m]):.4f}"
              f"  medAE {np.median(np.abs(f[m]-t[m])):.4f}  p90AE {np.percentile(np.abs(f[m]-t[m]),90):.4f}")
    for fld in ('chi2_per_dof','n_model_evals','seconds'):
        if fld in ok[0]:
            v=np.array([r[fld] for r in ok],float); v=v[np.isfinite(v)]
            if len(v): print(f"   {fld}: median {np.median(v):.4g}  mean {v.mean():.4g}")
    return out, ok

ref,_ = score('fits_refined_mu.json','refined (warm) fit')
cold,_ = score('fits_img.json','cold per-image fit')
net,_  = score('fits_pb3_mu.json','amortized network (B3 mu)')

print("\n--- ellipticity chain: net / cold / warm:",
      f"{net['e']:+.3f} / {cold['e']:+.3f} / {ref['e']:+.3f}")
print("improved on", sum(1 for k in KEYS if ref[k]>cold[k]), "of 7 vs cold")
for k in KEYS: print(f"   {k:9s} cold {cold[k]:+.3f} -> warm {ref[k]:+.3f}  net {net[k]:+.3f}")

# ---------------- part 2: spin-2, chi2 floor, ladder ----------------
import json, csv, numpy as np
from scipy.stats import spearmanr, mannwhitneyu
R="/mnt/user-data/uploads/Grid_Based_Experiment/superres/results"
MAN="/root/.claude/uploads/ff05c4ab-0730-58da-8f88-db5c381e7d59/75ce89d4-manifest.csv"
man={r['path'].replace('/', '\\'):r for r in csv.DictReader(open(MAN))}

# ---------- spin-2 shrinkage for the network ----------
net=json.load(open(f"{R}/fits_pb3_mu.json"))['rows']
cold=json.load(open(f"{R}/fits_img.json"))['rows']
e1p=np.array([r['e1'] for r in net]); e2p=np.array([r['e2'] for r in net])
T=[man[r['path'].replace('/', '\\')] for r in net]
e1t=np.array([float(r['host_e1']) for r in T]); e2t=np.array([float(r['host_e2']) for r in T])
mp=np.hypot(e1p,e2p); mt=np.hypot(e1t,e2t)
php=np.arctan2(e2p,e1p)/2; pht=np.arctan2(e2t,e1t)/2
dphi=np.degrees(np.abs(((php-pht+np.pi/2)%np.pi)-np.pi/2))
print("=== network spin-2")
print("  median |e| pred %.4f  true %.4f  overall ratio %.3f"%(np.median(mp),np.median(mt),np.median(mp)/np.median(mt)))
print("  median phase error %.1f deg (random = 45)"%np.median(dphi))
for lo,hi in [(0,5),(5,15),(15,30),(30,90)]:
    m=(dphi>=lo)&(dphi<hi)
    print(f"   {lo:2d}-{hi:2d} deg  n={m.sum():4d}  med|e|pred {np.median(mp[m]):.4f}  med|e|true {np.median(mt[m]):.4f}  ratio {np.median(mp[m])/np.median(mt[m]):.2f}")
e1c=np.array([r['e1'] for r in cold]); e2c=np.array([r['e2'] for r in cold])
phc=np.arctan2(e2c,e1c)/2
dphic=np.degrees(np.abs(((phc-pht+np.pi/2)%np.pi)-np.pi/2))
print("  cold fit: median phase err %.1f deg, modulus ratio %.2f"%(np.median(dphic), np.median(np.hypot(e1c,e2c))/np.median(mt)))

# ---------- chi2/dof by SNR bin, from b4 sersic rows ----------
b4=json.load(open(f"{R}/b4_sersic_metrics.json"))
rows=b4['rows']
ch=np.array([r['chi2_per_dof'] for r in rows]); sn=np.array([r['snr_max'] for r in rows])
q=np.percentile(sn,[33.33,66.67])
print("\n=== b4 sersic chi2/dof  median %.0f"%np.median(ch))
for i,(a,b) in enumerate([(-1,q[0]),(q[0],q[1]),(q[1],1e9)]):
    m=(sn>a)&(sn<=b); print(f"   SNR bin {i+1} ({sn[m].min():.1f}-{sn[m].max():.1f})  n={m.sum()}  median chi2/dof {np.median(ch[m]):.0f}")
print("   src_flux_in_box median %.3f  p10 %.3f"%(np.median([r['src_flux_in_box'] for r in rows]), np.percentile([r['src_flux_in_box'] for r in rows],10)))

# ---------- null baselines ----------
tc=json.load(open(f"{R}/truth_chi2.json"))
print("\n=== truth_chi2 keys:", [k for k in tc] if isinstance(tc,dict) else len(tc))
if isinstance(tc,dict):
    for k,v in tc.items():
        if k not in ('rows',): print("   ",k, json.dumps(v)[:600])
    if 'rows' in tc: print("   rows n=",len(tc['rows']), json.dumps(tc['rows'][0])[:600])

# ---------- superres metrics ----------
print("\n=== superres_metrics"); print(json.dumps(json.load(open(f"{R}/superres_metrics.json")), indent=1)[:1200])

# ---------- lens ladder ----------
ll=json.load(open(f"{R}/lens_ladder.json"))
print("\n=== lens_ladder keys:", list(ll.keys()) if isinstance(ll,dict) else len(ll))
if isinstance(ll,dict):
    for k,v in ll.items():
        if k!='rows': print("   ",k, json.dumps(v)[:800])

# ---------------- part 3: injection, table 1, magnification ----------------
import json, numpy as np
from scipy.stats import spearmanr, mannwhitneyu
R="/mnt/user-data/uploads/Grid_Based_Experiment/superres/results"

d=json.load(open(f"{R}/mu_resolution.json"))
print("=== mu_resolution keys:", list(d.keys()))
print("  ckpt", d['ckpt'], " fwhm", d['clump_fwhm'], " frac", d['clump_frac'], " bins", d['bins'])
ea=d['experiment_a']
print("  experiment_a (bin -> n, err):")
for k in sorted(ea,key=int):
    n,err=ea[k]; lo,hi=d['bins'][int(k)],d['bins'][int(k)+1]
    print(f"    mu {lo:g}-{hi:g}  n={n:4d}  err={err:.4f}")
rows=d['experiment_b']; print("  experiment_b n=",len(rows))
mu=np.array([r['mu'] for r in rows]); ct=np.array([r['contrast'] for r in rows]); sn=np.array([r['snr'] for r in rows])
print("  contrast: median %.3f  mean %.3f  p25 %.3f p75 %.3f"%(np.median(ct),ct.mean(),*np.percentile(ct,[25,75])))
lo=ct[mu<2]; hi=ct[mu>=2]
print("  mu<2 n=%d mean %.4f | mu>=2 n=%d mean %.4f | MW p=%.3g"%(len(lo),lo.mean(),len(hi),hi.mean(),mannwhitneyu(hi,lo,alternative='greater').pvalue))
print("  spearman contrast vs mu  %+.3f ; vs snr %+.3f"%(spearmanr(ct,mu).correlation, spearmanr(ct,sn).correlation))
qs=np.percentile(sn,[33.3,66.7])
for i,(a,b) in enumerate([(-1,qs[0]),(qs[0],qs[1]),(qs[1],1e9)]):
    m=(sn>a)&(sn<=b)
    r=ct[m&(mu>=2)].mean()/ct[m&(mu<2)].mean()
    print(f"   snr stratum {i+1}: ratio(mu>=2 / mu<2) = {r:.2f}   (n {m.sum()})")
# power law over populated bins
xs=[];ys=[]
for k in sorted(ea,key=int):
    n,err=ea[k]
    if n<20: continue
    lo_,hi_=d['bins'][int(k)],d['bins'][int(k)+1]; xs.append(np.sqrt(lo_*hi_)); ys.append(err)
sl=np.polyfit(np.log10(xs),np.log10(ys),1)[0]
print("  power-law slope of whole-source error vs mu: %.2f  over %d bins"%(sl,len(xs)))

print("\n=== B3 mu vs uniform (source-plane, corrected)")
for f in ('fits_pb3_mu.json','fits_pb3_uniform.json'):
    j=json.load(open(f"{R}/{f}"))
    print(" ",f, json.dumps(j['source_truth_corrected']))
    rr=j['rows']
    print("   corr_rms_frac med %.4f  corr_vs_mu med %.3f  saturation med %.3f"%(
        np.median([r['corr_rms_frac'] for r in rr]),
        np.median([r['corr_vs_mu'] for r in rr]),
        np.median([r.get('corr_saturation',0) for r in rr])))

print("\n=== B4 / B5 source-plane summary")
for f in ('b4_sersic_metrics.json','b4_sersic_seed1_metrics.json','b5_gate_metrics.json','b4_free_metrics.json'):
    j=json.load(open(f"{R}/{f}"))
    b=j['b4']; c=j['config']
    ch=np.array([r['chi2_per_dof'] for r in j['rows']])
    print(f"  {f:34s} ep={c['epochs']:3d} he={c['half_extent']} base={c['base']:6s} seed={c['seed']}"
          f"  corr {b['corr']:.4f} nmse {b['nmse']:.4f} size {b['size_ratio']:.4f} peak {b['peak_ratio']:.4f}"
          f"  chi2med {np.median(ch):.0f}")
print("  parametric (Sersic-only ceiling):", json.dumps(json.load(open(f"{R}/b4_sersic_metrics.json"))['parametric']))

print("\n=== magnification")
m=json.load(open(f"{R}/magnification.json"))['summary']
print(json.dumps(m, indent=1))
print("  rcrit rms in px:", m['rcrit_rms_err_arcsec']/0.10593)
