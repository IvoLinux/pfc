"""
Build an artificial LLM-friendly dataset from raw PCAP(s) + CICFlowMeter CSV.

Inputs:
  - One PCAP file (for now).
  - CICFlowMeter CSV (must have 'Timestamp', 'Flow Duration', 'Label' columns).

Output:
  - CSV with columns: Text, Label
    where Text = newline-separated header lines for all packets within the
    [Timestamp, Timestamp + FlowDuration] window.

Notes:
  - We store packet headers in a local SQLite DB (ts REAL, header TEXT) with an index on ts.
  - Flow Duration unit in CICFlowMeter is typically microseconds (µs). You can override.
  - Timestamp parsing tries ISO-like formats automatically (pandas) and assumes naive UTC
    unless you pass a timezone.

Usage example:
  python make_artificial_ds.py \
      --pcap /path/to/Friday-WorkingHours.pcap \
      --flows /path/to/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
      --out out_dataset.csv

"""

import argparse
import csv
import datetime as dt
import hashlib
import os
import sqlite3
import sys
from typing import Optional, Tuple

import dpkt
import pandas as pd

# ----------------------- Header rendering helpers -----------------------

def _ip_to_str(x: bytes) -> str:
    return ".".join(str(b) for b in x)

def _ipv6_to_str(x: bytes) -> str:
    # simple IPv6 render (no compression)
    return ":".join(f"{x[i]:02x}{x[i+1]:02x}" for i in range(0, 16, 2))

def render_header_line(ts: float, buf: bytes) -> Optional[str]:
    """
    Parse Ethernet/IP/(TCP|UDP|ICMP) and render a compact single-line, payload-free header string.
    Returns None if the packet is unparseable or unsupported.
    """
    try:
        eth = dpkt.ethernet.Ethernet(buf)

        # Base fields shared
        ts_int = f"{ts:.6f}"

        if isinstance(eth.data, dpkt.ip.IP):
            ip = eth.data
            src = _ip_to_str(ip.src)
            dst = _ip_to_str(ip.dst)
            ttl = getattr(ip, 'ttl', None)
            ip_len = getattr(ip, 'len', None)
            proto = ip.p
            l3 = f"ipv4 src={src} dst={dst} ttl={ttl} ip_len={ip_len} proto={proto}"

            # TCP
            if isinstance(ip.data, dpkt.tcp.TCP):
                tcp = ip.data
                flags = tcp.flags
                wnd = tcp.win
                optlen = len(tcp.opts) if tcp.opts else 0
                line = (
                    f"ts={ts_int} {l3} tcp sport={tcp.sport} dport={tcp.dport} "
                    f"flags={flags} win={wnd} optlen={optlen} hdrlen={(tcp.off*4)}"
                )
                return line

            # UDP
            if isinstance(ip.data, dpkt.udp.UDP):
                udp = ip.data
                line = (
                    f"ts={ts_int} {l3} udp sport={udp.sport} dport={udp.dport} "
                    f"ulen={udp.ulen}"
                )
                return line

            # ICMP
            if isinstance(ip.data, dpkt.icmp.ICMP):
                icmp = ip.data
                line = f"ts={ts_int} {l3} icmp type={icmp.type} code={icmp.code}"
                return line

            # Other L4
            return f"ts={ts_int} {l3} l4=other"

        elif isinstance(eth.data, dpkt.ip6.IP6):  # IPv6
            ip6 = eth.data
            src = _ipv6_to_str(ip6.src)
            dst = _ipv6_to_str(ip6.dst)
            hlim = getattr(ip6, 'hlim', None)
            plen = getattr(ip6, 'plen', None)
            nxt = ip6.nxt
            l3 = f"ipv6 src={src} dst={dst} hlim={hlim} plen={plen} nxt={nxt}"

            # TCP
            if isinstance(ip6.data, dpkt.tcp.TCP):
                tcp = ip6.data
                flags = tcp.flags
                wnd = tcp.win
                optlen = len(tcp.opts) if tcp.opts else 0
                line = (
                    f"ts={ts_int} {l3} tcp sport={tcp.sport} dport={tcp.dport} "
                    f"flags={flags} win={wnd} optlen={optlen} hdrlen={(tcp.off*4)}"
                )
                return line

            # UDP
            if isinstance(ip6.data, dpkt.udp.UDP):
                udp = ip6.data
                line = (
                    f"ts={ts_int} {l3} udp sport={udp.sport} dport={udp.dport} "
                    f"ulen={udp.ulen}"
                )
                return line

            # ICMPv6
            if isinstance(ip6.data, dpkt.icmp6.ICMP6):
                icmp6 = ip6.data
                line = f"ts={ts_int} {l3} icmp6 type={icmp6.type} code={icmp6.code}"
                return line

            return f"ts={ts_int} {l3} l4=other"

        else:
            # Non-IP (ARP, etc.). Keep minimal info so timing remains represented.
            return f"ts={ts_int} l2=non_ip"

    except Exception:
        # Corrupt or unparseable packet — skip quietly
        return None

# ----------------------- SQLite index -----------------------

def ensure_db(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS packets (
            ts REAL NOT NULL,
            header TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_packets_ts ON packets(ts)")
    con.commit()
    return con

def index_pcap_into_db(pcap_path: str, con: sqlite3.Connection, batch_size: int = 5000):
    """
    One pass over the pcap -> write (ts, header_line) rows to SQLite in batches.
    """
    inserted = 0
    cur = con.cursor()
    batch = []

    with open(pcap_path, "rb") as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except (ValueError, dpkt.dpkt.NeedData):
            # Try pcapng
            f.seek(0)
            pcap = dpkt.pcapng.Reader(f)

        for ts, buf in pcap:
            header_line = render_header_line(ts, buf)
            if header_line is None:
                continue
            batch.append((float(ts), header_line))
            if len(batch) >= batch_size:
                cur.executemany("INSERT INTO packets (ts, header) VALUES (?, ?)", batch)
                con.commit()
                inserted += len(batch)
                batch.clear()

        if batch:
            cur.executemany("INSERT INTO packets (ts, header) VALUES (?, ?)", batch)
            con.commit()
            inserted += len(batch)

    return inserted

# ----------------------- Time parsing -----------------------

def parse_flow_window(
    ts_str: str,
    duration_value: float,
    duration_unit: str = "us",
    tz: Optional[str] = None
) -> Tuple[float, float]:
    """
    Parse cicflowmeter row into [start_epoch, end_epoch] in seconds.
    - ts_str: cicflowmeter 'Timestamp' (varies by dataset formatting).
    - duration_value: numeric duration.
    - duration_unit: 'us' (microseconds), 'ms', or 's'.
    - tz: e.g., 'UTC' or 'America/Toronto'. If None, treat as naive UTC.
    """
    # Robust parse with pandas
    ts = pd.to_datetime(ts_str, errors="coerce", utc=(tz is None))
    if ts is pd.NaT:
        # Try common formats if needed (fallback)
        raise ValueError(f"Unparseable Timestamp: {ts_str}")

    if tz is not None:
        # localize-naive then convert to UTC
        ts = pd.to_datetime(ts_str).tz_localize(tz).tz_convert("UTC")

    start_epoch = ts.value / 1e9  # ns -> s

    # Duration to seconds
    if duration_unit == "us":
        dt_seconds = float(duration_value) / 1_000_000.0
    elif duration_unit == "ms":
        dt_seconds = float(duration_value) / 1_000.0
    else:
        dt_seconds = float(duration_value)

    end_epoch = start_epoch + dt_seconds
    return start_epoch, end_epoch

# ----------------------- Dataset builder -----------------------

def build_dataset_from_windows(
    con: sqlite3.Connection,
    flows_csv: str,
    out_csv: str,
    ts_col: str = "Timestamp",
    dur_col: str = "Flow Duration",
    label_col: str = "Label",
    duration_unit: str = "us",
    tz: Optional[str] = None,
    limit_rows: Optional[int] = None,
    dedup_identical_windows: bool = False
):
    """
    For each flow row, query [start, end] and emit a single textual sample + label.
    """
    df = pd.read_csv(flows_csv)

    required = [ts_col, dur_col, label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in flows CSV: {missing}")

    # Prepare writer
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out_f = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=["Text", "Label"])
    writer.writeheader()

    cur = con.cursor()
    seen_hashes = set()

    nrows = len(df) if limit_rows is None else min(limit_rows, len(df))
    for i in range(nrows):
        row = df.iloc[i]
        try:
            start_ts, end_ts = parse_flow_window(
                str(row[ts_col]),
                row[dur_col],
                duration_unit=duration_unit,
                tz=tz
            )
        except Exception as e:
            # Skip rows with invalid times
            continue

        # SQLite range query
        cur.execute(
            "SELECT header FROM packets WHERE ts >= ? AND ts <= ? ORDER BY ts ASC",
            (start_ts, end_ts),
        )
        headers = [h[0] for h in cur.fetchall()]
        if not headers:
            # No packets in window — keep an empty text or skip?
            # We'll keep empty to preserve label distribution; change if you prefer skipping.
            window_text = ""
        else:
            window_text = "\n".join(headers)

        if dedup_identical_windows and window_text:
            h = hashlib.sha256(window_text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                # Skip exact duplicate window (optional strategy)
                continue
            seen_hashes.add(h)

        writer.writerow({
            "Text": window_text,
            "Label": row[label_col]
        })

    out_f.close()

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Create artificial LLM dataset from PCAP + CICFlowMeter CSV.")
    ap.add_argument("--pcap", required=True, help="Path to a PCAP/PCAPNG file.")
    ap.add_argument("--flows", required=True, help="Path to CICFlowMeter CSV.")
    ap.add_argument("--out", required=True, help="Path to output CSV with Text,Label.")
    ap.add_argument("--db", default=None, help="Optional path to SQLite DB (default: alongside PCAP).")
    ap.add_argument("--ts-col", default="Timestamp", help="Timestamp column in flows CSV.")
    ap.add_argument("--dur-col", default="Flow Duration", help="Flow duration column in flows CSV.")
    ap.add_argument("--label-col", default="Label", help="Label column in flows CSV.")
    ap.add_argument("--duration-unit", choices=["us", "ms", "s"], default="us", help="Unit of Flow Duration.")
    ap.add_argument("--tz", default=None, help="Timezone of the CSV timestamps (e.g., 'UTC'). If omitted, treat as UTC.")
    # ap.add_argument("--limit-rows", type=int, default=None, help="Limit number of flow rows for a quick run.")
    ap.add_argument("--limit-rows", type=int, default=10, help="Limit number of flow rows for a quick run.")
    ap.add_argument("--skip-index", action="store_true", help="Assume DB already exists and skip re-indexing the PCAP.")
    ap.add_argument("--dedup-windows", action="store_true", help="Drop exact-duplicate windows by content hash.")
    args = ap.parse_args()

    db_path = args.db or (os.path.splitext(os.path.abspath(args.pcap))[0] + ".packets.sqlite")
    con = ensure_db(db_path)

    if not args.skip_index:
        print(f"[+] Indexing PCAP into {db_path} ...", file=sys.stderr)
        inserted = index_pcap_into_db(args.pcap, con)
        print(f"[+] Inserted {inserted} packet headers.", file=sys.stderr)
    else:
        print(f"[+] Skipping index build; using existing DB: {db_path}", file=sys.stderr)

    print(f"[+] Building dataset to {args.out} ...", file=sys.stderr)
    build_dataset_from_windows(
        con=con,
        flows_csv=args.flows,
        out_csv=args.out,
        ts_col=args.ts_col,
        dur_col=args.dur_col,
        label_col=args.label_col,
        duration_unit=args.duration_unit,
        tz=args.tz,
        limit_rows=args.limit_rows,
        dedup_identical_windows=args.dedup_windows
    )
    print("[+] Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
