"""Table 7 of the article: rho (spot-vol correlation) and nu (vol-of-vol) across underlyings.

Vol indices: Cboe daily history CSVs in data/ (VXAPL, VXAZN, VXGOG, VXGS, VXIBM, GVZ, OVX; VIX if present).
Spot: Nasdaq closes sampled every fifth trading day from 2021-09-01 (data/weekly_<TICKER>.csv, written by
fetch_data.py); Apple uses data/AAPL_close.csv (daily) sampled every fifth row so that all rows are comparable.
Weekly = consecutive sampled observations; monthly = every fourth (about 20 trading days); both non-overlapping.
Run:  python calibrate_multi.py [--tex]
"""
import pandas as pd, numpy as np, sys, pathlib

D = pathlib.Path("data")
PAIRS = [  # (label, spot file, vol file, vol column)
    ('Apple / VXAPL', 'AAPL_close.csv', 'VXAPL_History.csv', 'CLOSE'),
    ('Amazon / VXAZN', 'weekly_AMZN.csv', 'VXAZN_History.csv', 'CLOSE'),
    ('Alphabet / VXGOG', 'weekly_GOOG.csv', 'VXGOG_History.csv', 'CLOSE'),
    ('Goldman Sachs / VXGS', 'weekly_GS.csv', 'VXGS_History.csv', 'CLOSE'),
    ('IBM / VXIBM', 'weekly_IBM.csv', 'VXIBM_History.csv', 'CLOSE'),
    ('S\\&P 500 ETF (SPY) / VIX', 'weekly_SPY.csv', 'VIX_History.csv', 'CLOSE'),
    ('Gold ETF (GLD) / GVZ', 'weekly_GLD.csv', 'GVZ_History.csv', 'GVZ'),
    ('Oil ETF (USO) / OVX', 'weekly_USO.csv', 'OVX_History.csv', 'OVX'),
]

def load(spot, vol, col):
    vx = pd.read_csv(D / vol); vx['DATE'] = pd.to_datetime(vx['DATE'], format='%m/%d/%Y')
    vx = vx[['DATE', col]].rename(columns={col: 'VX'})
    sp = pd.read_csv(D / spot); sp['DATE'] = pd.to_datetime(sp['DATE']); sp = sp.rename(columns={'CLOSE': 'S'})
    sp = sp.sort_values('DATE'); sp = sp[sp.DATE >= '2021-09-01']
    if not spot.startswith('weekly_'): sp = sp.iloc[::5]          # daily file: sample every fifth trading day
    vxd = vx.set_index('DATE').sort_index().loc['2021-09-01':'2026-09-03', 'VX']   # full daily index for buckets
    return vx.merge(sp, on='DATE').sort_values('DATE').set_index('DATE').loc['2021-09-01':'2026-09-03'], vxd

def stats(df, vxd):
    lS, lV = np.log(df.S), np.log(df.VX); out = {}
    for name, h, ann in (('weekly', 1, 52), ('monthly', 4, 12)):
        dS = lS.diff(h).dropna().iloc[::h]; dV = lV.diff(h).dropna().iloc[::h]
        out[name] = (len(dS), np.corrcoef(dS, dV)[0, 1], dV.std() * np.sqrt(ann))
    out['buckets'] = ((vxd < 20).mean(), ((vxd >= 20) & (vxd < 35)).mean(), (vxd >= 35).mean())   # all trading days
    out['median_vx'] = vxd.median()
    return out

def table(tex=False):
    rows = []
    for label, spot, vol, col in PAIRS:
        try: df, vxd = load(spot, vol, col)
        except FileNotFoundError as e:
            print(f"{label:26s} skipped ({e.filename} not found)"); continue
        st = stats(df, vxd); nw, rw, vw = st['weekly']; nm, rm, vm = st['monthly']; b = st['buckets']
        rows.append((label, rw, vw, rm, vm, st['median_vx'], b))
        if tex:
            print(f"{label} & ${rw:+.2f}$ & {vw:.2f} & ${rm:+.2f}$ & {vm:.2f} & {st['median_vx']:.0f}\\% & "
                  f"{round(100*b[0])} / {round(100*b[1])} / {round(100*b[2])} \\\\")
        else:
            print(f"{label:26s} weekly n={nw:3d} rho={rw:+.2f} nu={vw:.2f} | monthly n={nm:3d} rho={rm:+.2f} nu={vm:.2f} "
                  f"| median {st['median_vx']:.1f} | buckets {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f}")
    return rows

if __name__ == '__main__':
    table(tex='--tex' in sys.argv)
