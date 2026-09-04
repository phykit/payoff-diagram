import numpy as np, matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from model import *

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{mathpazo}\usepackage{textcomp}\usepackage{amsmath}",
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.7, "axes.edgecolor": "#4a4a4a", "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a", "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE="#1f5f9e"; GREY="#6a6a6a"; ORANGE="#d1590a"; INK="#1a1a1a"
S = np.linspace(30,70,801)
GBP = FuncFormatter(lambda v,_: (r"$-$" if v<0 else "")+f"{abs(v):.0f}")
def style(ax, xlab="Share price", ylab=None):
    ax.axhline(0, color="#8a8a8a", lw=0.7, zorder=1); ax.set_xlabel(xlab)
    if ylab: ax.set_ylabel(ylab)
    ax.yaxis.set_major_formatter(GBP); ax.xaxis.set_major_formatter(GBP); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
def head(ax, title, sub):
    ax.set_title(title, loc="left", fontsize=10, pad=24)
    ax.annotate(sub, xy=(0,1.02), xycoords="axes fraction", fontsize=8, color="#5a5a5a")

C0 = float(bs(S0,K,T,SIG0,r,"c"))
# ---- Fig 1
fig, ax = plt.subplots(figsize=(6.4,3.7))
for tau,c,l in zip([0.5,0.25,6/52,1/52], ["#c6dbef","#8fbadd","#4a8fc4","#1f5f9e"], ["6 months left","3 months left","6 weeks left","1 week left"]):
    ax.plot(S, bs(S,K,tau,SIG0,r,"c")-C0, color=c, lw=1.6, label=l, zorder=3)
ax.plot(S, np.maximum(S-K,0.0)-C0, color=INK, lw=1.4, ls=(0,(4,2)), label="Expiry payoff (the diagram you were shown)", zorder=4)
style(ax, ylab="Profit / loss if unwound"); ax.set_xlim(30,70); ax.set_ylim(-6,21)
head(ax, "The position you hold is a curve, not a hockey stick", r"Long 50-strike call, six-month tenor, 25\% vol, premium 4.00")
ax.legend(loc="upper left"); fig.savefig("v2_fig1.pdf"); plt.close(fig)

# ---- Fig 2
tau2 = T - 21/250; cols = {0.15:BLUE, 0.25:GREY, 0.45:ORANGE}
fig, ax = plt.subplots(figsize=(6.4,3.7))
for sig in REGIMES:
    ax.plot(S, straddle(S,tau2,sig)-STR0, color=cols[sig], lw=1.8, zorder=3,
            label={0.15:r"Vol crushed to 15\%",0.25:r"Vol unchanged (25\%)",0.45:r"Vol spikes to 45\%"}[sig])
y45 = straddle(S,tau2,0.45)-STR0; imin=int(np.argmin(y45))
V45 = straddle(S,tau2,0.45)
for sp,ls,lab in ((0.02,(0,(1,1.5)),r"45\%, net of a 2\% spread"),(0.10,(0,(3,1.5)),r"45\%, net of a 10\% spread")):
    ax.plot(S, V45*(1-sp/2) - STR0*(1+sp/2), color=ORANGE, lw=1.1, ls=ls, zorder=3, label=lab)
ax.plot(S, np.maximum(S-K,0)+np.maximum(K-S,0)-STR0, color=INK, lw=1.2, ls=(0,(4,2)), label="Expiry payoff", zorder=2)
ax.plot(S[imin], y45[imin], "o", ms=5, color=ORANGE, mec="white", mew=1.0, zorder=5)
ax.annotate(f"worst case: $+${y45[imin]:.2f}", xy=(S[imin],y45[imin]), xytext=(0,-13), textcoords="offset points", ha="center", fontsize=8, color=ORANGE)
style(ax, xlab="Share price three weeks after opening the trade", ylab="Profit / loss if unwound")
ax.set_xlim(30,70); ax.set_ylim(-9,17.5)
head(ax, "What a volatility spike does to a six-month straddle", r"Long 50-strike straddle bought at 25\% vol for 7.02; unwound three weeks in")
ax.legend(loc="upper center", ncol=2, fontsize=7.3); fig.savefig("v2_fig2.pdf"); plt.close(fig)

# ---- Fig 3 fan (unconditional lognormal, 20% vol)
sig3=0.20; t=np.linspace(0,0.5,400); med=S0*np.exp((r-0.5*sig3**2)*t); sd=sig3*np.sqrt(t)
fig, ax = plt.subplots(figsize=(6.4,3.7))
for z,a,lab in [(3,0.10,"99.7"),(2,0.16,"95"),(1,0.26,"68")]:
    ax.fill_between(t*12, med*np.exp(-z*sd), med*np.exp(z*sd), color=BLUE, alpha=a, lw=0, zorder=2, label=lab+r"\% of outcomes")
ax.plot(t*12, med, color=BLUE, lw=1.4, zorder=4, label="Median path")
for lvl,txt in ((57.0,"upper break-even 57"),(43.0,"lower break-even 43")):
    ax.axhline(lvl, color=INK, lw=0.9, ls=(0,(4,2)), zorder=5); ax.annotate(txt, xy=(0.15,lvl), xytext=(0.15,lvl+0.9), fontsize=8)
ax.set_xlim(0,6); ax.set_ylim(30,79); ax.set_xlabel("Months from today"); ax.set_ylabel("Share price")
ax.yaxis.set_major_formatter(GBP); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
head(ax, "Quiet stocks are not where you think they are", r"A 50 share at 20\% vol, $\mu = r = 4\%$: lognormal price envelope, with the straddle's break-evens overlaid")
ax.legend(loc="lower left", ncol=2); fig.savefig("v2_fig3.pdf"); plt.close(fig)

# ---- Fig 4: the grid, revised (conditional bands + cell statistics)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
rows=[0.125,0.25,0.375,0.5]; rowlab={0.125:"6 weeks in\n$u=T/4$",0.25:"3 months in\n$u=T/2$",0.375:"4.5 months in\n$u=3T/4$",0.5:"at expiry\n$u=T$"}
fig, axes = plt.subplots(4,3, figsize=(6.05,8.6), sharex=True, sharey=True)
lo=hi=0
for te in rows:
    for sg in REGIMES:
        y=straddle(S,T-te,sg)-STR0; lo=min(lo,y.min()); hi=max(hi,y.max())
for i,te in enumerate(rows):
    for j,sg in enumerate(REGIMES):
        ax=axes[i,j]; m,s_=cond_lognormal(te,sg); P,E,Q=cell_stats(te,sg); pr=regime_prob(te,sg)
        for z,a in ((2,0.10),(1,0.17)): ax.axvspan(np.exp(m-z*s_), np.exp(m+z*s_), color=BLUE, alpha=a, lw=0, zorder=1)
        ax.axvline(np.exp(m), color=BLUE, lw=0.7, alpha=0.7, zorder=2)
        ax.axhline(0, color="#8a8a8a", lw=0.6, zorder=2)
        ax.plot(S, straddle(S,T-te,sg)-STR0, color=INK, lw=1.4, zorder=3)
        ax.set_xlim(30,70); ax.set_ylim(lo-1.5, hi+8.5)
        ax.grid(True, color="#e2e2e2", lw=0.4); ax.set_axisbelow(True)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        gbp=lambda v: ("$-$" if v<0 else "$+$")+f"{abs(v):.2f}"
        ax.text(0.03,0.97, f"regime {pr:.0%}\nprofit {P:.0%}   E {gbp(E)}   $Q_{{5\\%}}$ {gbp(Q)}",
                transform=ax.transAxes, fontsize=7.4, va="top", ha="left", color="#222222", linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), zorder=6)
        if i==0: ax.set_title(r"$\sigma_I = $"+f"{int(sg*100)}"+r"\%"+("  (base case)" if sg==SIG0 else ""), fontsize=9, pad=6,
                              fontweight=("bold" if sg==SIG0 else "normal"))
        if j==0: ax.set_ylabel(rowlab[te], fontsize=8.2)
        if j==1 and sg==SIG0:
            for sp in ("left","bottom"): ax.spines[sp].set_linewidth(1.2)
        ax.tick_params(labelsize=7); ax.yaxis.set_major_formatter(GBP); ax.xaxis.set_major_formatter(GBP)
fig.supxlabel("Share price at the unwind date", fontsize=8.5, y=0.045)
fig.supylabel("Profit / loss per share if unwound (currency units); one contract = 100 shares", fontsize=8.5, x=0.012)
handles=[Line2D([0],[0],color=INK,lw=1.4,label=r"Unwind P\&L at $\sigma_I$ (mid-market)"),
         Patch(facecolor=BLUE,alpha=0.27,label=r"68\% band of spot, given the regime"),
         Patch(facecolor=BLUE,alpha=0.12,label=r"95\% band"),
         Line2D([0],[0],color=BLUE,lw=0.7,label="Conditional median of spot")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.3, frameon=False, bbox_to_anchor=(0.5,0.0))
fig.suptitle("One trade, twelve honest pictures", x=0.02, ha="left", fontsize=10.5, y=0.985)
fig.text(0.02,0.958, "Long 50-strike straddle bought at 25\\% vol: P\\&L if unwound (black) at each quarter of its life, by prevailing implied-vol regime $\\sigma_I$.\nBands: where the share is likely to be given that regime, with realised vol $\\sigma_R=\\sigma_I$ (lognormal; $\\rho="+f"{RHO:.2f}"+"$, $\\nu="+f"{NU:.2f}"+"$ from Apple, 2021 to 2026).",
         fontsize=7.6, color="#5a5a5a", va="top")
fig.tight_layout(rect=(0.02,0.055,1,0.925)); fig.savefig("v2_fig4.pdf"); plt.close(fig)

# ---- Fig 5: quantile-dotplot variant, one row (3 months in)
te=0.25; NQ=20
fig, axes = plt.subplots(1,3, figsize=(6.8,2.9), sharey=True)
for j,sg in enumerate(REGIMES):
    ax=axes[j]; m,s=cond_lognormal(te,sg); tau=T-te
    q=np.exp(m + s*norm.ppf((np.arange(NQ)+0.5)/NQ))
    pnl_q = straddle(q,tau,sg)-STR0
    y=straddle(S,tau,sg)-STR0
    ax.axhline(0,color="#8a8a8a",lw=0.6,zorder=2); ax.plot(S,y,color=INK,lw=1.4,zorder=3)
    # stack dots in bins of width 2 along the x axis at the bottom of the panel
    base=-8.5; edges=np.arange(30,72,2); counts={}
    for qq,pp in zip(q,pnl_q):
        b=int((qq-30)//2); k=counts.get(b,0); counts[b]=k+1
        ax.plot(30+2*b+1, base+k*1.15, "o", ms=4.2, color=(BLUE if pp>0 else ORANGE), mec="white", mew=0.5, zorder=5, clip_on=False)
    nwin=int((pnl_q>0).sum())
    ax.text(0.03,0.96, f"{nwin} of {NQ} outcomes in profit", transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=6)
    ax.set_xlim(30,70); ax.set_ylim(-9.5, 16); ax.set_title(f"{int(sg*100)}"+r"\% vol regime", fontsize=9, pad=6)
    ax.grid(True,color="#e2e2e2",lw=0.4); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7); ax.yaxis.set_major_formatter(GBP); ax.xaxis.set_major_formatter(GBP)
    ax.set_xlabel("Share price", fontsize=8)
axes[0].set_ylabel("3 months in", fontsize=8.5)
fig.suptitle("The same row, frequency-framed", x=0.02, ha="left", fontsize=10.5, y=1.04)
fig.text(0.02,0.975, "Each dot is one of twenty equally likely share prices three months in, under the cell's regime; blue dots unwind at a profit, orange at a loss.",
         fontsize=7.6, color="#5a5a5a", va="top")
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig("v2_fig5.pdf"); plt.close(fig)

# ---- Fig 6: 0DTE grid (index, points), rho=0 by construction
I0=100.0; T0=1/252; regs0=[0.12,0.18,0.30]; hours=[1.0,3.0,5.0,6.5]
X=np.linspace(95,105,801); PTS=FuncFormatter(lambda v,_: (r"$-$" if v<0 else "")+f"{abs(v):g}")
prem0=float(straddle(I0,T0,0.18,K=I0))
fig, axes = plt.subplots(4,3, figsize=(6.8,7.6), sharex=True, sharey=True)
for i,h in enumerate(hours):
    te=T0*h/6.5; tau=T0-te
    for j,sg in enumerate(regs0):
        ax=axes[i,j]; sd=sg*np.sqrt(te); m=np.log(I0)+(r-0.5*sg**2)*te
        y=straddle(X,tau,sg,K=I0)-prem0
        for z,a in ((2,0.10),(1,0.17)): ax.axvspan(np.exp(m-z*sd),np.exp(m+z*sd),color=BLUE,alpha=a,lw=0,zorder=1)
        ax.axhline(0,color="#8a8a8a",lw=0.6,zorder=2); ax.plot(X,y,color=INK,lw=1.4,zorder=3)
        # cell stats
        xs=np.linspace(m-8*sd,m+8*sd,4001); dens=norm.pdf((xs-m)/sd)/sd; p=straddle(np.exp(xs),tau,sg,K=I0)-prem0
        P=np.trapezoid(dens*(p>0),xs); E=np.trapezoid(dens*p,xs)
        ax.text(0.03,0.96, f"P(profit) {P:.0%}, E[P\\&L] {'$-$' if E<0 else '$+$'}{abs(E):.2f}", transform=ax.transAxes, fontsize=6.6, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=6)
        ax.set_xlim(95,105); ax.set_ylim(-1.3,4.6); ax.grid(True,color="#e2e2e2",lw=0.4); ax.set_axisbelow(True)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        if i==0: ax.set_title(f"{int(sg*100)}"+r"\% vol regime", fontsize=9, pad=6)
        if j==0: ax.set_ylabel({1.0:"1 hour in",3.0:"3 hours in",5.0:"5 hours in",6.5:"at the close"}[h], fontsize=8.5)
        ax.tick_params(labelsize=7); ax.yaxis.set_major_formatter(PTS); ax.xaxis.set_major_formatter(PTS)
for j in range(3): axes[3,j].set_xlabel("Index level (points)", fontsize=8)
fig.suptitle("The same display at zero days to expiry", x=0.02, ha="left", fontsize=10.5, y=0.985)
fig.text(0.02,0.958, f"Long at-the-money straddle on an index at 100, bought at the open at 18\\% vol for {prem0:.2f} points, 6.5 trading hours to expiry.\nBands: 68\\% and 95\\% lognormal ranges for the index by that hour under the cell's vol; no spot--vol correlation (see text).",
         fontsize=7.6, color="#5a5a5a", va="top")
fig.tight_layout(rect=(0,0,1,0.925)); fig.savefig("v2_fig6.pdf"); plt.close(fig)
print("0DTE premium", round(prem0,3)); print("done")

# ---- Fig 7: robustness across strategies, one row (u = T/2), three regimes ----
te=0.25; tau=T-te
C0=float(bs(S0,K,T,SIG0,r,"c")); P45_0=float(bs(S0,45,T,SIG0,r,"p")); P40_0=float(bs(S0,40,T,SIG0,r,"p")); CR0=P45_0-P40_0
strats=[("Long 50 call (debit 4.00)", lambda x,sg: bs(np.exp(x),K,tau,sg,r,"c")-C0),
        (f"Short 45/40 put spread (credit {CR0:.2f})", lambda x,sg: CR0-(bs(np.exp(x),45,tau,sg,r,"p")-bs(np.exp(x),40,tau,sg,r,"p"))),
        ("Short 50 straddle (credit 7.02)", lambda x,sg: STR0-straddle(np.exp(x),tau,sg))]
fig, axes = plt.subplots(3,3, figsize=(6.05,6.6), sharex=True, sharey="row")
for i,(name,fn) in enumerate(strats):
    for j,sg in enumerate(REGIMES):
        ax=axes[i,j]; m,s_=cond_lognormal(te,sg)
        P,E,Q=cell_stats(te,sg,pnl_fn=lambda x,sg=sg,fn=fn: fn(x,sg)); pr=regime_prob(te,sg)
        for z,a in ((2,0.10),(1,0.17)): ax.axvspan(np.exp(m-z*s_), np.exp(m+z*s_), color=BLUE, alpha=a, lw=0, zorder=1)
        ax.axvline(np.exp(m), color=BLUE, lw=0.7, alpha=0.7, zorder=2); ax.axhline(0, color="#8a8a8a", lw=0.6, zorder=2)
        ax.plot(S, fn(np.log(S),sg), color=INK, lw=1.4, zorder=3)
        ax.plot(S, fn(np.log(S),1e-9) if False else fn(np.log(S),sg), color=INK, lw=0, zorder=0)
        ax.grid(True, color="#e2e2e2", lw=0.4); ax.set_axisbelow(True)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        gbp=lambda v: ("$-$" if v<0 else "$+$")+f"{abs(v):.2f}"
        ax.text(0.03,0.97, f"regime {pr:.0%}\nprofit {P:.0%}   E {gbp(E)}   $Q_{{5\\%}}$ {gbp(Q)}", transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
                color="#222222", linespacing=1.25, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), zorder=6)
        if i==0: ax.set_title(r"$\sigma_I = $"+f"{int(sg*100)}"+r"\%", fontsize=9, pad=6, fontweight=("bold" if sg==SIG0 else "normal"))
        if j==0: ax.set_ylabel(name, fontsize=7.6)
        ax.set_xlim(30,70); ax.tick_params(labelsize=7); ax.yaxis.set_major_formatter(GBP); ax.xaxis.set_major_formatter(GBP)
    ymin=min(fn(np.log(S),sg).min() for sg in REGIMES); ymax=max(fn(np.log(S),sg).max() for sg in REGIMES); pad=(ymax-ymin)
    axes[i,0].set_ylim(ymin-0.08*pad, ymax+0.55*pad)
fig.supxlabel("Share price three months in", fontsize=8.5, y=0.02)
fig.suptitle("The same row for three other positions", x=0.02, ha="left", fontsize=10.5, y=0.985)
fig.text(0.02,0.955, "Unwind P\\&L per share at $u = T/2$ by implied-vol regime, with the conditional 68\\% and 95\\% bands and cell statistics of Figure 4.\nThe short straddle is the mirror of Figure 4: the display shows losses where the terminal diagram shows a plateau.", fontsize=7.6, color="#5a5a5a", va="top")
fig.tight_layout(rect=(0,0.03,1,0.925)); fig.savefig("v2_fig7.pdf"); plt.close(fig)
print("fig7 credit", round(CR0,3))
