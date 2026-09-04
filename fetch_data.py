"""Acquire the calibration data. Cboe VXAPL is a public download; Nasdaq closes are fetched from Nasdaq's
public API and are NOT redistributed with this package. Writes data/VXAPL_History.csv and data/AAPL_close.csv."""
import json, urllib.request, csv, hashlib, sys, pathlib
d = pathlib.Path("data"); d.mkdir(exist_ok=True)
vx = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXAPL_History.csv"
urllib.request.urlretrieve(vx, d/"VXAPL_History.csv")
req = urllib.request.Request("https://api.nasdaq.com/api/quote/AAPL/historical?assetclass=stocks&fromdate=2021-09-01&limit=9999&todate=2026-09-04",
                             headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
rows = json.load(urllib.request.urlopen(req))["data"]["tradesTable"]["rows"]
with open(d/"AAPL_close.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["DATE","CLOSE"])
    for r in reversed(rows):
        m,dd,y = r["date"].split("/"); w.writerow([f"{y}-{m}-{dd}", r["close"].replace("$","")])
for p in ("VXAPL_History.csv","AAPL_close.csv"):
    print(p, hashlib.md5(open(d/p,"rb").read()).hexdigest())
print("Compare AAPL_close.csv against checksums.txt (per-year row counts and close sums); VXAPL grows daily, so compare the 2021-09-01 to 2026-09-03 slice only.")
