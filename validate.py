"""Prints every number quoted in the article and checks them against expected_numbers.json.
Run:  python validate.py        (exit code 0 = all match within tolerance)"""
import json, sys, numpy as np
from scipy.stats import norm
from model import *

out = {}
out["call"] = float(bs(S0,K,T,SIG0,r,"c")); out["put"] = float(bs(S0,K,T,SIG0,r,"p")); out["straddle"] = STR0
S = np.linspace(30,70,801); tau2 = T-21/250
y45 = straddle(S,tau2,0.45)-STR0; i = int(np.argmin(y45))
out["fig2_min"] = float(y45[i]); out["fig2_min_spot"] = float(S[i]); out["fig2_at_spot50"] = float(straddle(50,tau2,0.45)-STR0)
out["table1_theta"] = float(straddle(S0,tau2,SIG0)-STR0); out["table1_vega"] = float(straddle(S0,tau2,0.45)-straddle(S0,tau2,SIG0))
Vt = float(straddle(S0,tau2,0.45))
for sp in (0.02,0.05,0.10,0.20): out[f"table2_roundtrip_{int(sp*100)}"] = sp/2*STR0 + sp/2*Vt
for sp in (0.02,0.10):
    net = straddle(S,tau2,0.45)*(1-sp/2) - STR0*(1+sp/2); out[f"fig2_net_floor_{int(sp*100)}"] = float(net.min())
sig3=0.20; sd=sig3*np.sqrt(0.5); med=S0*np.exp((r-0.5*sig3**2)*0.5)
out["fig3_95_lo"] = float(med*np.exp(-2*sd)); out["fig3_95_hi"] = float(med*np.exp(2*sd))
for te,name in ((0.125,"6w"),(0.5,"expiry")):
    for sg in REGIMES:
        P,E,Q = cell_stats(te,sg); pr = regime_prob(te,sg); m,s = cond_lognormal(te,sg)
        out[f"grid_{name}_{int(sg*100)}_Pregime"] = pr; out[f"grid_{name}_{int(sg*100)}_Pprofit"] = P
        out[f"grid_{name}_{int(sg*100)}_E"] = E; out[f"grid_{name}_{int(sg*100)}_Q5"] = Q; out[f"grid_{name}_{int(sg*100)}_centre"] = float(np.exp(m))
te=0.25; NQ=20
for sg in REGIMES:
    m,s = cond_lognormal(te,sg); q = np.exp(m+s*norm.ppf((np.arange(NQ)+0.5)/NQ)); out[f"dots_{int(sg*100)}"] = int((straddle(q,T-te,sg)-STR0>0).sum())
# 0DTE
I0=100.0; T0=1/252; prem0=float(straddle(I0,T0,0.18,K=I0)); out["odte_premium"]=prem0
X=np.linspace(95,105,801)
for sg in (0.12,0.18,0.30):
    for h,name in ((1.0,"1h"),(6.5,"close")):
        te=T0*h/6.5; sd=sg*np.sqrt(te); m=np.log(I0)+(r-0.5*sg**2)*te
        xs=np.linspace(m-8*sd,m+8*sd,4001); dens=norm.pdf((xs-m)/sd)/sd; p=straddle(np.exp(xs),T0-te,sg,K=I0)-prem0
        out[f"odte_{int(sg*100)}_{name}_Pprofit"]=float(np.trapezoid(dens*(p>0),xs)); out[f"odte_{int(sg*100)}_{name}_E"]=float(np.trapezoid(dens*p,xs))
# Figure 6 (robustness row, u = T/2)
te=0.25; tau=T-te
C0=float(bs(S0,K,T,SIG0,r,"c")); CR0=float(bs(S0,45,T,SIG0,r,"p")-bs(S0,40,T,SIG0,r,"p")); out["fig6_putspread_credit"]=CR0
fns={"call":lambda x,sg: bs(np.exp(x),K,tau,sg,r,"c")-C0,
     "putspread":lambda x,sg: CR0-(bs(np.exp(x),45,tau,sg,r,"p")-bs(np.exp(x),40,tau,sg,r,"p")),
     "shortstraddle":lambda x,sg: STR0-straddle(np.exp(x),tau,sg)}
for nm,fn in fns.items():
    for sg in REGIMES:
        P,E,Q=cell_stats(te,sg,pnl_fn=lambda x,sg=sg,fn=fn: fn(x,sg)); out[f"fig6_{nm}_{int(sg*100)}_Pprofit"]=P; out[f"fig6_{nm}_{int(sg*100)}_E"]=E; out[f"fig6_{nm}_{int(sg*100)}_Q5"]=Q
# Table 6 (OU sensitivity, Appendix A)
for sg in REGIMES:
    out[f"ou_expiry_{int(sg*100)}_Pregime"]=regime_prob_ou(0.5,sg); P,E,c=cell_stats_ou(0.5,sg); out[f"ou_expiry_{int(sg*100)}_Pprofit"]=P; out[f"ou_expiry_{int(sg*100)}_centre"]=c
# Table 4 (robustness of the bottom row) and the drift sensitivity of Appendix A
import robustness
for label, cells in robustness.table().items():
    key = label.split(',')[0].split(':')[0].replace(' ','_').replace('$','').replace('\\','').lower()[:24]
    for sg,(pr,P,E) in zip(REGIMES,cells):
        out[f"t5_{key}_{int(sg*100)}_Pregime"]=pr; out[f"t5_{key}_{int(sg*100)}_Pprofit"]=P; out[f"t5_{key}_{int(sg*100)}_E"]=E
for mu in (0.0,0.04,0.10,0.15):
    P,E,Q=cell_stats(0.5,0.25,mu=mu); out[f"drift_{int(mu*100)}_Pprofit"]=P; out[f"drift_{int(mu*100)}_E"]=E
# Monte Carlo cross-check
P,E,Q = cell_stats(0.25,0.45); Pm,Em,Qm = mc_check(0.25,0.45)
out["mc_dP"]=abs(P-Pm); out["mc_dE"]=abs(E-Em); out["mc_dQ"]=abs(Q-Qm)
# bucket-mean alternative
for te,name in ((0.125,"6w"),(0.5,"expiry")):
    zpt=np.log(0.45/SIG0)/(NU*np.sqrt(te)); zb=bucket_mean_z(0.45,te); m_pt,_=cond_lognormal(te,0.45)
    out[f"bucketmean_centre_{name}"]=float(np.exp(m_pt - RHO*0.45*np.sqrt(te)*zpt + RHO*0.45*np.sqrt(te)*zb))

for k,v in out.items(): print(f"{k:32s} {v:12.4f}" if isinstance(v,float) else f"{k:32s} {v}")
try:
    exp = json.load(open("expected_numbers.json"))
    bad = [(k, v, exp[k]) for k,v in out.items() if k in exp and abs(float(v)-float(exp[k])) > (0.03 if k.startswith("mc_") else 0.0105)]
    print("\nCHECK:", "PASS" if not bad else f"FAIL {bad}")
    sys.exit(1 if bad else 0)
except FileNotFoundError:
    json.dump(out, open("expected_numbers.json","w"), indent=1); print("\nexpected_numbers.json written")
