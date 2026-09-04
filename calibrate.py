"""Estimates rho (spot-vol correlation) and nu (vol-of-vol) from data/VXAPL_History.csv and data/AAPL_close.csv
over 2021-09-01 to 2026-09-03, at daily, weekly, monthly and quarterly non-overlapping horizons."""
import pandas as pd, numpy as np
vx=pd.read_csv('data/VXAPL_History.csv'); vx['DATE']=pd.to_datetime(vx['DATE'],format='%m/%d/%Y'); vx=vx[['DATE','CLOSE']].rename(columns={'CLOSE':'VX'})
ap=pd.read_csv('data/AAPL_close.csv'); ap['DATE']=pd.to_datetime(ap['DATE']); ap=ap.rename(columns={'CLOSE':'S'})
df=vx.merge(ap,on='DATE').sort_values('DATE').set_index('DATE').loc['2021-09-01':'2026-09-03']
lS=np.log(df.S); lV=np.log(df.VX)
for h,name in ((1,'daily'),(5,'weekly'),(21,'monthly'),(63,'quarterly')):
    dS=lS.diff(h).dropna().iloc[::h]; dV=lV.diff(h).dropna().iloc[::h]
    print(f"{name:9s} n={len(dS):4d} rho={np.corrcoef(dS,dV)[0,1]:+.2f} nu={dV.std()*np.sqrt(252/h):.2f}")
print("bucket freq <20 / 20-35 / >35:", f"{(df.VX<20).mean():.2f} {((df.VX>=20)&(df.VX<35)).mean():.2f} {(df.VX>=35).mean():.2f}")
d21S=lS.diff(21); d21V=lV.diff(21); m=d21V>np.log(45/25); print("21d windows with vol up >=80%:", int(m.sum()), "mean spot return", f"{d21S[m].mean():+.3f}")
