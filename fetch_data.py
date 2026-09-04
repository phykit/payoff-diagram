"""Acquire the calibration data.

Cboe implied-volatility index histories (VXAPL, VXAZN, VXGOG, VXGS, VXIBM, GVZ, OVX) are public downloads.
Nasdaq closes are fetched from Nasdaq's public API and are NOT redistributed with this package.

Writes  data/<INDEX>_History.csv            (Cboe, daily)
        data/AAPL_close.csv                 (Nasdaq, daily; used by calibrate.py)
        data/weekly_<TICKER>.csv            (Nasdaq closes sampled every fifth trading day from 2021-09-01;
                                             used by calibrate_multi.py for AMZN, GOOG, GS, IBM, GLD, USO, SPY)
"""
import json, urllib.request, csv, hashlib, pathlib
d = pathlib.Path("data"); d.mkdir(exist_ok=True)
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
for idx in ("VXAPL", "VXAZN", "VXGOG", "VXGS", "VXIBM", "GVZ", "OVX"):
    urllib.request.urlretrieve(CBOE.format(idx), d / f"{idx}_History.csv")

def nasdaq(ticker, assetclass="stocks"):
    url = (f"https://api.nasdaq.com/api/quote/{ticker}/historical?assetclass={assetclass}"
           f"&fromdate=2021-09-01&limit=9999&todate=2026-09-04")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    rows = json.load(urllib.request.urlopen(req))["data"]["tradesTable"]["rows"]
    out = []
    for r in reversed(rows):                      # oldest first
        m, dd, y = r["date"].split("/"); out.append((f"{y}-{m}-{dd}", float(r["close"].replace("$", "").replace(",", ""))))
    return out

def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["DATE", "CLOSE"]); w.writerows(rows)

write(d / "AAPL_close.csv", nasdaq("AAPL"))
for t, ac in (("AMZN", "stocks"), ("GOOG", "stocks"), ("GS", "stocks"), ("IBM", "stocks"),
              ("GLD", "etf"), ("USO", "etf"), ("SPY", "etf")):
    daily = nasdaq(t, ac); weekly = daily[::5]     # every fifth trading day, 2021-09-01 first
    write(d / f"weekly_{t}.csv", weekly)
    n = len(weekly); s = sum(c for _, c in weekly); sd = sum(int(dt.replace("-", "")) for dt, _ in weekly)
    print(f"weekly_{t}.csv n={n} sum={s:.3f} sumdate={sd}   (compare with checksums.txt)")
for p in ("VXAPL_History.csv", "AAPL_close.csv"):
    print(p, hashlib.md5(open(d / p, "rb").read()).hexdigest())
print("AAPL: compare against checksums.txt (per-year row counts and close sums); Cboe files grow daily, so compare the 2021-09-01 to 2026-09-03 slice only.")
