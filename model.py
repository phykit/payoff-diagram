"""Shared model for the revised article: pricing, joint spot-vol measure, cell statistics."""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

S0 = K = 50.0; r = 0.04; SIG0 = 0.25; T = 0.5
NU = 0.85    # annualised vol-of-vol: Apple, VXAPL monthly changes, Sep 2021 to Sep 2026
RHO = -0.60  # spot-vol correlation: Apple, AAPL vs VXAPL, Sep 2021 to Sep 2026
REGIMES = [0.15, 0.25, 0.45]
BUCKETS = [(0.0, 0.20), (0.20, 0.35), (0.35, np.inf)]   # regime probability buckets

def bs(S, K, tau, sig, r, kind):
    S = np.asarray(S, dtype=float)
    if tau <= 1e-12:
        return np.maximum(S-K, 0.0) if kind == "c" else np.maximum(K-S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sig**2)*tau)/(sig*np.sqrt(tau)); d2 = d1 - sig*np.sqrt(tau)
    if kind == "c": return S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    return K*np.exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)

def straddle(S, tau, sig, K=K, r=r): return bs(S, K, tau, sig, r, "c") + bs(S, K, tau, sig, r, "p")

STR0 = float(straddle(S0, T, SIG0))

def cond_lognormal(t, sig_I, sig_R=None, rho=RHO, nu=NU, mu=r, S0=S0, sig0=SIG0):
    """Parameters (m, s) of ln S_t | vol regime sig_I at time t.
    sig_R: realised vol over [0,t] for the band width (default: tracks the regime)."""
    if sig_R is None: sig_R = sig_I
    z = np.log(sig_I/sig0)/(nu*np.sqrt(t)) if t > 0 else 0.0
    m = np.log(S0) + (mu - 0.5*sig_R**2)*t + rho*sig_R*np.sqrt(t)*z
    s = sig_R*np.sqrt(t)*np.sqrt(1 - rho**2)
    return m, s

def regime_prob(t, sig_I, nu=NU, sig0=SIG0):
    """P(sigma_t in the bucket containing sig_I), ln sigma_t ~ N(ln sig0, nu^2 t)."""
    if t <= 0: return 1.0 if sig_I == sig0 else 0.0
    lo, hi = next(b for b in BUCKETS if b[0] <= sig_I < b[1])
    sd = nu*np.sqrt(t)
    F = lambda x: norm.cdf((np.log(x/sig0))/sd) if np.isfinite(x) else 1.0
    return F(hi) - (F(lo) if lo > 0 else 0.0)

def cell_stats(t, sig_I, sig_R=None, rho=RHO, nu=NU, q=0.05, pnl_fn=None, mu=r):
    """P(P&L>0), E[P&L] and the q-quantile of P&L for a position unwound at elapsed time t
    under the cell's conditional measure. Adaptive quadrature over +/- 8 conditional s.d.
    (scipy.integrate.quad, epsabs=1e-9); the quantile is taken on a 20,001-point
    probability-weighted grid of the same range. See mc_check() for the Monte Carlo cross-check.
    mu: drift of ln S for the band (default r, the risk-neutral drift; the display's Table 4 varies it)."""
    m, s = cond_lognormal(t, sig_I, sig_R, rho, nu, mu=mu)
    tau = T - t
    pnl = pnl_fn or (lambda x: straddle(np.exp(x), tau, sig_I) - STR0)
    dens = lambda x: norm.pdf((x-m)/s)/s
    E = quad(lambda x: pnl(x)*dens(x), m-8*s, m+8*s, limit=200, epsabs=1e-9)[0]
    P = quad(lambda x: dens(x)*(pnl(x) > 0), m-8*s, m+8*s, limit=400, points=[m], epsabs=1e-9)[0]
    xs = np.linspace(m-8*s, m+8*s, 20001); w = dens(xs); w /= w.sum()
    p = pnl(xs); o = np.argsort(p); cw = np.cumsum(w[o]); Q = p[o][np.searchsorted(cw, q)]
    return P, E, Q

def mc_check(t, sig_I, n=400000, seed=1):
    """Monte Carlo cross-check of cell_stats under the same conditional lognormal."""
    m, s = cond_lognormal(t, sig_I); rng = np.random.default_rng(seed)
    x = rng.normal(m, s, n); p = straddle(np.exp(x), T-t, sig_I) - STR0
    return (p > 0).mean(), p.mean(), np.quantile(p, 0.05)

def bucket_mean_z(sig_I, t, nu=NU, sig0=SIG0):
    """E[z | bucket] for the truncated-normal alternative to the point-value shift convention."""
    lo, hi = next(b for b in BUCKETS if b[0] <= sig_I < b[1]); sd = nu*np.sqrt(t)
    a = np.log(lo/sig0)/sd if lo > 0 else -np.inf; b = np.log(hi/sig0)/sd if np.isfinite(hi) else np.inf
    Z = norm.cdf(b) - norm.cdf(a)
    return (norm.pdf(a if np.isfinite(a) else 0)*(np.isfinite(a)) - norm.pdf(b if np.isfinite(b) else 0)*(np.isfinite(b)))/Z

if __name__ == "__main__":
    print(f"straddle premium {STR0:.4f}")
    tau2 = T - 21/250
    v_theta = straddle(S0, tau2, SIG0) - STR0
    v_vega  = straddle(S0, tau2, 0.45) - straddle(S0, tau2, SIG0)
    total   = straddle(S0, tau2, 0.45) - STR0
    # first-order vega x dSigma
    h=1e-4; vega = (straddle(S0,tau2,SIG0+h)-straddle(S0,tau2,SIG0-h))/(2*h)
    print(f"theta {v_theta:+.2f}  vega-regime {v_vega:+.2f}  total {total:+.2f}  vega/1vol {vega/100:.3f}  first-order {vega*0.20:+.2f}  convexity {v_vega-vega*0.20:+.2f}")
    # frictions
    Vt = straddle(S0, tau2, 0.45)
    for sp in (0.02, 0.05, 0.10, 0.20):
        cost = sp/2*STR0 + sp/2*Vt
        print(f"spread {sp:.0%}: round trip {cost:.2f} = {cost/4.21:.0%} of the 4.21 floor; net at spot {total-cost:+.2f}")
    print("grid stats:")
    for t in (0.125,0.25,0.375,0.5):
        row=[]
        for sI in REGIMES:
            P,E,Q = cell_stats(t,sI); pr = regime_prob(t,sI); m,s = cond_lognormal(t,sI)
            row.append(f"σ={sI:.0%}: P(reg)={pr:.2f} centre={np.exp(m):.1f} P(win)={P:.2f} E={E:+.2f} Q5={Q:+.2f}")
        print(f" t={t}: " + " | ".join(row))

    print("MC cross-check (t=0.25, 45%):", cell_stats(0.25,0.45), mc_check(0.25,0.45))
    print("MC cross-check (t=0.5, 25%):", cell_stats(0.5,0.25), mc_check(0.5,0.25))
    for t in (0.125,0.5):
        zpt = np.log(0.45/SIG0)/(NU*np.sqrt(t)); zb = bucket_mean_z(0.45,t)
        m_pt,_ = cond_lognormal(t,0.45); m_b = m_pt - RHO*0.45*np.sqrt(t)*zpt + RHO*0.45*np.sqrt(t)*zb
        print(f"t={t}: point z={zpt:.2f} centre={np.exp(m_pt):.1f}; bucket-mean z={zb:.2f} centre={np.exp(m_b):.1f}")

# ---- Sensitivity: mean-reverting (Ornstein-Uhlenbeck) log-vol, W-15 ----
KAPPA = 6.9   # per year, Apple monthly AR(1) on ln VXAPL, Sep 2021 to Sep 2026 (half-life ~1.2 months)

def ou_sd(t, kappa=KAPPA, nu=NU):
    return nu*np.sqrt((1-np.exp(-2*kappa*t))/(2*kappa))

def regime_prob_ou(t, sig_I, kappa=KAPPA, nu=NU, sig0=SIG0, sigbar=None):
    """P(bucket) when ln sigma_t is OU about ln sigbar (default sigma_0), started at ln sigma_0."""
    sigbar = sig0 if sigbar is None else sigbar
    lo, hi = next(b for b in BUCKETS if b[0] <= sig_I < b[1]); sd = ou_sd(t, kappa, nu)
    mean = np.log(sigbar) + (np.log(sig0)-np.log(sigbar))*np.exp(-kappa*t)
    F = lambda x: norm.cdf((np.log(x)-mean)/sd) if np.isfinite(x) else 1.0
    return F(hi) - (F(lo) if lo > 0 else 0.0)

def cell_stats_ou(t, sig_I, kappa=KAPPA, rho=RHO, nu=NU):
    """Cell statistics with the conditional shift computed from the OU standardisation of the vol move."""
    z = np.log(sig_I/SIG0)/ou_sd(t, kappa, nu)
    sig_R = sig_I
    m = np.log(S0) + (r - 0.5*sig_R**2)*t + rho*sig_R*np.sqrt(t)*z
    s = sig_R*np.sqrt(t)*np.sqrt(1-rho**2)
    tau = T - t; pnl = lambda x: straddle(np.exp(x), tau, sig_I) - STR0
    dens = lambda x: norm.pdf((x-m)/s)/s
    E = quad(lambda x: pnl(x)*dens(x), m-8*s, m+8*s, limit=200)[0]
    P = quad(lambda x: dens(x)*(pnl(x) > 0), m-8*s, m+8*s, limit=400, points=[m])[0]
    return P, E, float(np.exp(m))
