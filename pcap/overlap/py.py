import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # if running headless (e.g., server/WSL/cron)

OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREFIX = "a"   # or "1402" — anything you like

# --- CONFIG ---
csv_path = "2302.csv"
timestamp_col = "Timestamp"          # parse format flexibly; set dayfirst=True for 14/02
flow_dur_col = "Flow Duration"       # microseconds
fwd_pkts_col = "Tot Fwd Pkts"
bwd_pkts_col = "Tot Bwd Pkts"
# Optional bytes:
fwd_bytes_col = "TotLen Fwd Pkts"
bwd_bytes_col = "TotLen Bwd Pkts"

# --- LOAD ---
df = pd.read_csv(csv_path)

# Parse times (CIC sometimes has DD/MM/YYYY HH:MM:SS format)
df["start"] = pd.to_datetime(df[timestamp_col], errors="coerce", dayfirst=True)
dur_s = pd.to_numeric(df[flow_dur_col], errors="coerce").fillna(0) / 1_000_000.0
df["end"] = df["start"] + pd.to_timedelta(dur_s, unit="s")

# Clean
m = df["start"].notna() & df["end"].notna() & (df["end"] > df["start"])
df = df.loc[m].copy()
df["dur_s"] = np.maximum((df["end"] - df["start"]).dt.total_seconds(), 1)

# Packet/byte rates
tot_pkts = pd.to_numeric(df[fwd_pkts_col], errors="coerce").fillna(0) + \
           pd.to_numeric(df[bwd_pkts_col], errors="coerce").fillna(0)
df["pps"] = np.where(df["dur_s"] > 0, tot_pkts / df["dur_s"], 0)

# If bytes available:
if {fwd_bytes_col, bwd_bytes_col}.issubset(df.columns):
    tot_bytes = pd.to_numeric(df[fwd_bytes_col], errors="coerce").fillna(0) + \
                pd.to_numeric(df[bwd_bytes_col], errors="coerce").fillna(0)
    df["bps"] = np.where(df["dur_s"] > 0, (tot_bytes * 8) / df["dur_s"], 0.0)
else:
    df["bps"] = np.nan  # optional

# --- SWEEP EVENTS (unweighted and weighted) ---
events = pd.concat([
    pd.DataFrame({"t": df["start"], "d1": 1,  "dpps": df["pps"],  "dbps": df["bps"]}),
    pd.DataFrame({"t": df["end"],   "d1": -1, "dpps": -df["pps"], "dbps": -df["bps"]}),
], ignore_index=True)

events = events.sort_values("t")

# 1) Collapse duplicate timestamps by summing deltas at the same 't'
agg = events.groupby("t", as_index=True).agg({"d1":"sum", "dpps":"sum", "dbps":"sum"})

# 2) Now take cumulative sums (unique index -> safe for resample)
series_conc = agg["d1"].cumsum()
series_pps  = agg["dpps"].cumsum()
series_bps  = agg["dbps"].cumsum() if agg["dbps"].notna().any() else None

# 3) Resample for plotting (1-second buckets)
conc_1s = series_conc.resample("1s").ffill().fillna(0)
pps_1s  = series_pps.resample("1s").ffill().fillna(0)
bps_1s  = series_bps.resample("1s").ffill().fillna(0) if series_bps is not None else None


conc_1s.to_frame(name="concurrent_flows").to_csv(OUT_DIR / f"{PREFIX}_concurrent_flows_1s.csv")
pps_1s.to_frame(name="pps").to_csv(OUT_DIR / f"{PREFIX}_pps_1s.csv")
if bps_1s is not None:
    bps_1s.to_frame(name="bps").to_csv(OUT_DIR / f"{PREFIX}_bps_1s.csv")


# Quick scalars
avg_conc = conc_1s.mean()
p95_conc = conc_1s.quantile(0.95)
max_conc = conc_1s.max()

avg_pps = pps_1s.mean()
p95_pps = pps_1s.quantile(0.95)
max_pps = pps_1s.max()

import matplotlib.pyplot as plt

plt.figure()
conc_1s.plot()
plt.title("Concurrent Flows (1s)")
plt.xlabel("Time")
plt.ylabel("Flows")
plt.tight_layout()
plt.savefig(OUT_DIR / f"{PREFIX}_concurrent_flows_1s.png", dpi=150, bbox_inches="tight")

plt.figure()
pps_1s.plot()
plt.title("Traffic Intensity (pps, 1s)")
plt.xlabel("Time")
plt.ylabel("Packets per second")
plt.tight_layout()
plt.savefig(OUT_DIR / f"{PREFIX}_traffic_intensity_pps_1s.png", dpi=150, bbox_inches="tight")

if bps_1s is not None:
    plt.figure()
    bps_1s.plot()
    plt.title("Traffic Intensity (bps, 1s)")
    plt.xlabel("Time")
    plt.ylabel("bits per second")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{PREFIX}_traffic_intensity_bps_1s.png", dpi=150, bbox_inches="tight")

summary = pd.DataFrame({
    "metric": ["avg_conc","p95_conc","max_conc","avg_pps","p95_pps","max_pps"] + (["avg_bps","p95_bps","max_bps"] if bps_1s is not None else []),
    "value":  [avg_conc, p95_conc, max_conc, avg_pps, p95_pps, max_pps] + (
              [bps_1s.mean(), bps_1s.quantile(0.95), bps_1s.max()] if bps_1s is not None else [])
})
summary.to_csv(OUT_DIR / f"{PREFIX}_summary.csv", index=False)
summary.to_json(OUT_DIR / f"{PREFIX}_summary.json", orient="records", indent=2)
