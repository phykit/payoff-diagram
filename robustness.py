"""Table 4 of the article: robustness of the bottom row (at expiry) of Figure 4 to the modelling choices.

Each row changes one thing relative to the display's base case (lognormal vol law, rho = -0.6, nu = 0.85,
mu = r, mid-market) and reports, for the 15%, 25% and 45% columns, the regime probability, the probability
of an unwind at a profit and the expected P&L per share.
"""
import numpy as np
import model as m

T = m.T
COLS = m.REGIMES

def spread_pnl(tau, sig_I, s):
    """Net-of-spread P&L: entry at V0(1+s/2), exit at V_u(1-s/2)."""
    return lambda x: m.straddle(np.exp(x), tau, sig_I) * (1 - s / 2) - m.STR0 * (1 + s / 2)

def row_lognormal(rho, nu, mu=m.r, spread=0.0, t=T):
    out = []
    for sI in COLS:
        pr = m.regime_prob(t, sI, nu=nu)
        fn = spread_pnl(T - t, sI, spread) if spread else None
        P, E, Q = m.cell_stats(t, sI, rho=rho, nu=nu, mu=mu, pnl_fn=fn)
        out.append((pr, P, E))
    return out

def row_ou(t=T):
    out = []
    for sI in COLS:
        pr = m.regime_prob_ou(t, sI)
        P, E, med = m.cell_stats_ou(t, sI)
        out.append((pr, P, E))
    return out

ROWS = [
    ('Base case: lognormal vol, $\\rho=-0.6$, $\\nu=0.85$, $\\mu=r$, mid-market', lambda: row_lognormal(-0.6, 0.85)),
    ('Independent columns, $\\rho = 0$', lambda: row_lognormal(0.0, 0.85)),
    ('Mean-reverting vol law, $\\kappa = 6.9$ (Appendix~A)', row_ou),
    ('Physical drift, $\\mu = 10\\%$', lambda: row_lognormal(-0.6, 0.85, mu=0.10)),
    ('Round-trip spread of 10\\% of mid', lambda: row_lognormal(-0.6, 0.85, spread=0.10)),
    ('Goldman Sachs parameters, $\\rho=-0.59$, $\\nu=0.73$', lambda: row_lognormal(-0.59, 0.73)),
    ('Alphabet parameters, $\\rho=-0.28$, $\\nu=0.88$', lambda: row_lognormal(-0.28, 0.88)),
    ('Gold ETF parameters, $\\rho=+0.48$, $\\nu=0.52$', lambda: row_lognormal(0.48, 0.52)),
]

def table(tex=False):
    results = {}
    for label, fn in ROWS:
        cells = fn(); results[label] = cells
        if tex:
            pr = ' & '.join(f"{c[0]*100:.0f}\\%" for c in cells)
            pp = ' & '.join(f"{c[1]*100:.0f}\\%" for c in cells)
            ee = ' & '.join(f"${c[2]:+.2f}$" for c in cells)
            print(f"{label} & {pr} & {pp} & {ee} \\\\")
        else:
            print(f"{label[:60]:60s} " + ' | '.join(f"{c[0]:.2f} {c[1]:.2f} {c[2]:+.2f}" for c in cells))
    return results

if __name__ == '__main__':
    import sys
    table(tex='--tex' in sys.argv)
