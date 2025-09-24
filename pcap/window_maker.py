#!/usr/bin/env python3
"""
PCAP → SQLite (headers only) → CSV windows (Text,Label) for CICFlowMeter rows.

- DB stores "true" header state: structured L2/L3/L4 fields + a canonical header text.
- No payloads are stored.
- You can later change text rendering/masking without re-indexing.

Usage examples:

# Single file
py make_ds.py \
  --pcap ~/Desktop/RAW/DoS-02-16/pcap/capDESKTOP-AN3U28N-172.31.64.17 \
  --flows ~/Desktop/RAW-csv/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv \
  --out test.csv \
  --tz America/Moncton

# Whole folder (streaming)
py make_ds.py \
  --pcap-dir ~/Desktop/RAW/DoS-02-16/pcap \
  --flows ~/Desktop/RAW-csv/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv \
  --out friday16.csv \
  --tz America/Moncton
"""

import argparse, csv, os, sys, sqlite3, hashlib
from typing import Optional, Tuple, List
from datetime import datetime

import dpkt
import pandas as pd
import time
import struct

# ----------------------- Header parsing -----------------------

def _ip4(x: bytes) -> str:
    return ".".join(str(b) for b in x)

def _ip6(x: bytes) -> str:
    return ":".join(f"{x[i]:02x}{x[i+1]:02x}" for i in range(0, 16, 2))

def parse_packet(buf: bytes, ts: float):
    """
    Return a tuple: (fields_dict, header_text)
    fields_dict contains structured header facts (for DB),
    header_text is the canonical line used for Text in the CSV.
    """
    try:
        eth = dpkt.ethernet.Ethernet(buf)
    except Exception:
        return None, None

    ts_s = f"{ts:.6f}"

    # Defaults
    fields = dict(
        ts=float(ts),
        l2="ethernet",
        ip_version=None,
        src=None, dst=None,
        ttl=None, ip_len=None, ip_proto=None,
        hlim=None, plen=None, nxt=None,
        l4=None,
        sport=None, dport=None,
        tcp_flags=None, tcp_win=None, tcp_optlen=None, tcp_hdrlen=None,
        udp_ulen=None,
        icmp_type=None, icmp_code=None,
        icmp6_type=None, icmp6_code=None,
    )

    # Non-IP (ARP, etc.)
    if not isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
        header_text = f"ts={ts_s} l2=non_ip"
        return fields, header_text

    # IPv4
    if isinstance(eth.data, dpkt.ip.IP):
        ip = eth.data
        fields["ip_version"] = 4
        fields["src"] = _ip4(ip.src)
        fields["dst"] = _ip4(ip.dst)
        fields["ttl"] = getattr(ip, "ttl", None)
        fields["ip_len"] = getattr(ip, "len", None)
        fields["ip_proto"] = ip.p

        l3 = f"ipv4 src={fields['src']} dst={fields['dst']} ttl={fields['ttl']} ip_len={fields['ip_len']} proto={fields['ip_proto']}"

        # TCP
        if isinstance(ip.data, dpkt.tcp.TCP):
            tcp = ip.data
            fields.update(
                l4="tcp",
                sport=tcp.sport, dport=tcp.dport,
                tcp_flags=tcp.flags, tcp_win=tcp.win,
                tcp_optlen=len(tcp.opts) if tcp.opts else 0,
                tcp_hdrlen=(tcp.off * 4)
            )
            header_text = (
                f"ts={ts_s} {l3} tcp sport={tcp.sport} dport={tcp.dport} "
                f"flags={tcp.flags} win={tcp.win} optlen={fields['tcp_optlen']} hdrlen={fields['tcp_hdrlen']}"
            )
            return fields, header_text

        # UDP
        if isinstance(ip.data, dpkt.udp.UDP):
            udp = ip.data
            fields.update(l4="udp", sport=udp.sport, dport=udp.dport, udp_ulen=udp.ulen)
            header_text = (
                f"ts={ts_s} {l3} udp sport={udp.sport} dport={udp.dport} ulen={udp.ulen}"
            )
            return fields, header_text

        # ICMP
        if isinstance(ip.data, dpkt.icmp.ICMP):
            icmp = ip.data
            fields.update(l4="icmp", icmp_type=icmp.type, icmp_code=icmp.code)
            header_text = f"ts={ts_s} {l3} icmp type={icmp.type} code={icmp.code}"
            return fields, header_text

        # Other L4
        fields["l4"] = "other"
        header_text = f"ts={ts_s} {l3} l4=other"
        return fields, header_text

    # IPv6
    if isinstance(eth.data, dpkt.ip6.IP6):
        ip6 = eth.data
        fields["ip_version"] = 6
        fields["src"] = _ip6(ip6.src)
        fields["dst"] = _ip6(ip6.dst)
        fields["hlim"] = getattr(ip6, "hlim", None)
        fields["plen"] = getattr(ip6, "plen", None)
        fields["nxt"]  = ip6.nxt

        l3 = f"ipv6 src={fields['src']} dst={fields['dst']} hlim={fields['hlim']} plen={fields['plen']} nxt={fields['nxt']}"

        # TCP
        if isinstance(ip6.data, dpkt.tcp.TCP):
            tcp = ip6.data
            fields.update(
                l4="tcp",
                sport=tcp.sport, dport=tcp.dport,
                tcp_flags=tcp.flags, tcp_win=tcp.win,
                tcp_optlen=len(tcp.opts) if tcp.opts else 0,
                tcp_hdrlen=(tcp.off * 4)
            )
            header_text = (
                f"ts={ts_s} {l3} tcp sport={tcp.sport} dport={tcp.dport} "
                f"flags={tcp.flags} win={tcp.win} optlen={fields['tcp_optlen']} hdrlen={fields['tcp_hdrlen']}"
            )
            return fields, header_text

        # UDP
        if isinstance(ip6.data, dpkt.udp.UDP):
            udp = ip6.data
            fields.update(l4="udp", sport=udp.sport, dport=udp.dport, udp_ulen=udp.ulen)
            header_text = (
                f"ts={ts_s} {l3} udp sport={udp.sport} dport={udp.dport} ulen={udp.ulen}"
            )
            return fields, header_text

        # ICMPv6
        if isinstance(ip6.data, dpkt.icmp6.ICMP6):
            icmp6 = ip6.data
            fields.update(l4="icmp6", icmp6_type=icmp6.type, icmp6_code=icmp6.code)
            header_text = f"ts={ts_s} {l3} icmp6 type={icmp6.type} code={icmp6.code}"
            return fields, header_text

        # Other L4
        fields["l4"] = "other"
        header_text = f"ts={ts_s} {l3} l4=other"
        return fields, header_text

    # Shouldn’t get here
    return None, None

_PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1",  # PCAP little-endian, microsecond
    b"\xa1\xb2\xc3\xd4",  # PCAP big-endian, microsecond
    b"\x4d\x3c\xb2\xa1",  # PCAP little-endian, nanosecond
    b"\xa1\xb2\x3c\x4d",  # PCAP big-endian, nanosecond
}
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

def _looks_like_pcap_or_pcapng(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        if not head or len(head) < 4:
            return False
        return (head in _PCAP_MAGICS) or (head == _PCAPNG_MAGIC)
    except Exception:
        return False

# ----------------------- SQLite -----------------------

def ensure_db(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS packets (
            ts           REAL NOT NULL,
            l2           TEXT,
            ip_version   INTEGER,
            src          TEXT,
            dst          TEXT,
            ttl          INTEGER,
            ip_len       INTEGER,
            ip_proto     INTEGER,
            hlim         INTEGER,
            plen         INTEGER,
            nxt          INTEGER,
            l4           TEXT,
            sport        INTEGER,
            dport        INTEGER,
            tcp_flags    INTEGER,
            tcp_win      INTEGER,
            tcp_optlen   INTEGER,
            tcp_hdrlen   INTEGER,
            udp_ulen     INTEGER,
            icmp_type    INTEGER,
            icmp_code    INTEGER,
            icmp6_type   INTEGER,
            icmp6_code   INTEGER,
            header       TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_packets_ts ON packets(ts)")

    # NEW: track which files were ingested (path + size + mtime as the signature)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            path        TEXT PRIMARY KEY,
            size_bytes  INTEGER NOT NULL,
            mtime_ns    INTEGER NOT NULL,
            inserted    INTEGER NOT NULL,
            first_ts    REAL,
            last_ts     REAL,
            ingested_at TEXT NOT NULL
        )
    """)
    con.commit()
    return con

def _file_signature(path: str):
    st = os.stat(path)
    return (os.path.abspath(path), int(st.st_size), int(st.st_mtime_ns))

def already_ingested(con: sqlite3.Connection, path: str) -> Optional[tuple]:
    cur = con.cursor()
    cur.execute("SELECT size_bytes, mtime_ns FROM ingested_files WHERE path = ?", (os.path.abspath(path),))
    return cur.fetchone()  # (size_bytes, mtime_ns) or None

def mark_ingested(con: sqlite3.Connection, path: str, size_bytes: int, mtime_ns: int,
                  inserted: int, first_ts: Optional[float], last_ts: Optional[float]):
    cur = con.cursor()
    cur.execute("""
        INSERT INTO ingested_files (path, size_bytes, mtime_ns, inserted, first_ts, last_ts, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            size_bytes=excluded.size_bytes,
            mtime_ns=excluded.mtime_ns,
            inserted=excluded.inserted,
            first_ts=excluded.first_ts,
            last_ts=excluded.last_ts,
            ingested_at=excluded.ingested_at
    """, (os.path.abspath(path), size_bytes, mtime_ns, inserted, first_ts, last_ts, datetime.utcnow().isoformat() + "Z"))
    con.commit()

def insert_rows(cur, rows):
    cur.executemany("""
        INSERT INTO packets (
            ts,l2,ip_version,src,dst,ttl,ip_len,ip_proto,hlim,plen,nxt,l4,
            sport,dport,tcp_flags,tcp_win,tcp_optlen,tcp_hdrlen,udp_ulen,
            icmp_type,icmp_code,icmp6_type,icmp6_code,header
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)

def index_one_pcap(pcap_path: str, con: sqlite3.Connection, batch_size: int = 5000):
    cur = con.cursor()
    batch = []; inserted = 0
    first_ts = None
    last_ts = None

    with open(pcap_path, "rb") as f:
        # Try PCAP then PCAPNG
        try:
            reader = dpkt.pcap.Reader(f)
        except (ValueError, dpkt.dpkt.NeedData):
            f.seek(0)
            try:
                reader = dpkt.pcapng.Reader(f)
            except (ValueError, dpkt.dpkt.NeedData) as e:
                raise ValueError(f"invalid pcap/pcapng: {pcap_path}") from e

        try:
            for rec in reader:
                try:
                    ts, buf = rec
                except Exception:
                    continue  # ignore non-packet sections in pcapng
                try:
                    fields, header = parse_packet(buf, ts)
                except Exception:
                    continue

                if header is None:
                    continue

                if first_ts is None: first_ts = float(ts)
                last_ts = float(ts)

                row = (
                    fields["ts"],
                    fields["l2"],
                    fields["ip_version"],
                    fields["src"],
                    fields["dst"],
                    fields["ttl"],
                    fields["ip_len"],
                    fields["ip_proto"],
                    fields["hlim"],
                    fields["plen"],
                    fields["nxt"],
                    fields["l4"],
                    fields["sport"],
                    fields["dport"],
                    fields["tcp_flags"],
                    fields["tcp_win"],
                    fields["tcp_optlen"],
                    fields["tcp_hdrlen"],
                    fields["udp_ulen"],
                    fields["icmp_type"],
                    fields["icmp_code"],
                    fields["icmp6_type"],
                    fields["icmp6_code"],
                    header
                )
                batch.append(row)
                if len(batch) >= batch_size:
                    insert_rows(cur, batch); con.commit(); inserted += len(batch); batch.clear()

        except dpkt.dpkt.NeedData:
            print(f"[!] Truncated capture (NeedData): {pcap_path}", file=sys.stderr)
        except struct.error as e:
            print(f"[!] Corrupt record (struct.error) in {pcap_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[!] Error while reading {pcap_path}: {e.__class__.__name__}: {e}", file=sys.stderr)

    if batch:
        insert_rows(cur, batch); con.commit(); inserted += len(batch)
    return inserted, first_ts, last_ts

def index_dir(pcap_dir: str, con: sqlite3.Connection, batch_size: int = 5000, exts=(".pcap",".pcapng",".cap"), reindex: bool=False):
    total = 0
    SKIP_EXTS = {".sqlite", ".db", ".csv", ".parquet", ".json", ".txt", ".log"}
    for root, _, files in os.walk(pcap_dir):
        for name in files:
            path = os.path.join(root, name)
            lower = name.lower()
            _, ext = os.path.splitext(lower)
            if ext in SKIP_EXTS:
                continue

            # Allow by extension or by magic sniff
            allow = (exts is not None and lower.endswith(exts)) or _looks_like_pcap_or_pcapng(path)
            if not allow:
                continue

            # Skip if already ingested with same signature (unless --reindex)
            try:
                abspath, size_bytes, mtime_ns = _file_signature(path)
                prev = already_ingested(con, abspath)
                if prev and not reindex:
                    prev_size, prev_mtime_ns = prev
                    if prev_size == size_bytes and prev_mtime_ns == mtime_ns:
                        print(f"[i] Skipping (already ingested, unchanged): {path}", file=sys.stderr)
                        continue
            except Exception as e:
                print(f"[!] Could not stat {path}: {e}", file=sys.stderr)

            try:
                print(f"[+] Indexing {path}", file=sys.stderr)
                inserted, first_ts, last_ts = index_one_pcap(path, con, batch_size)
                # Record ingestion signature (even if 0 packets parsed)
                mark_ingested(con, abspath, size_bytes, mtime_ns, inserted, first_ts, last_ts)
                total += inserted
            except Exception as e:
                print(f"[!] Skipping {path}: {e.__class__.__name__}: {e}", file=sys.stderr)
                continue
    return total

def db_time_range(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM packets")
    row = cur.fetchone()
    if row and row[0] is not None:
        return row[0], row[1], row[2]
    return None, None, 0

# ----------------------- Time windows -----------------------

def parse_flow_window(ts_str: str, duration_value: float, duration_unit: str="us", tz: Optional[str]=None)->Tuple[float,float]:
    # CICFlowMeter timestamps are local (Atlantic for CICIDS2018). We localize then convert to UTC.
    if tz is None:
        ts = pd.to_datetime(ts_str, errors="coerce", utc=True, dayfirst=True)
    else:
        ts_local = pd.to_datetime(ts_str, errors="coerce", dayfirst=True)
        if ts_local is pd.NaT:
            raise ValueError(f"Unparseable Timestamp: {ts_str}")
        ts = ts_local.tz_localize(tz).tz_convert("UTC")
    start = ts.value / 1e9
    if duration_unit == "us":  dt = float(duration_value)/1_000_000.0
    elif duration_unit == "ms": dt = float(duration_value)/1_000.0
    else:                       dt = float(duration_value)
    return start, start + dt

def _parse_start_ts(ts_str: str, tz: Optional[str]) -> float:
    """
    Parse the CSV timestamp and return UTC epoch seconds (float), matching
    the timezone handling used elsewhere.
    """
    if tz is None:
        ts = pd.to_datetime(ts_str, errors="coerce", utc=True, dayfirst=True)
    else:
        ts_local = pd.to_datetime(ts_str, errors="coerce", dayfirst=True)
        if pd.isna(ts_local):
            raise ValueError(f"Unparseable Timestamp: {ts_str}")
        ts = ts_local.tz_localize(tz).tz_convert("UTC")
    return ts.value / 1e9

def print_db_time_range(con: sqlite3.Connection, label: str = "DB"):
    tmin, tmax, n = db_time_range(con)
    if n:
        print(f"[i] {label} packets: {n} rows", file=sys.stderr)
        print(f"[i] {label} ts range (UTC): {pd.to_datetime(tmin, unit='s', utc=True)} .. {pd.to_datetime(tmax, unit='s', utc=True)}", file=sys.stderr)
    else:
        print(f"[i] {label}: no rows", file=sys.stderr)

# ----------------------- Dataset builder -----------------------

def build_dataset_from_windows(
    con: sqlite3.Connection, flows_csv: str, out_csv: str,
    ts_col="Timestamp", dur_col="Flow Duration", label_col="Label",
    duration_unit="us", tz: Optional[str]=None,
    limit_rows: Optional[int]=None, dedup_identical_windows: bool=False,
    pkt_window: int = 32,   # smaller default is DistilBERT-friendly
):
    import re

    # Select the original canonical header text only
    SELECT_COL = "header"

    # Removal of noisy / token-heavy fields:
    # - timestamps, IP addresses, low-level IP fields, TCP optlen/hdrlen
    # Keep: l3/l4 words (ipv4/ipv6, tcp/udp/icmp), dport/sport, flags, win, udp_ulen, icmp types/codes, etc.
    BAD_FIELDS_RE = re.compile(
        r'\b(?:'
        r'ts|src|dst|ttl|ip_proto|hlim|nxt|optlen|hdrlen'
        r')=[^\s]+'  # eat "key=value" up to next space
    )
    MULTISPACE_RE = re.compile(r'\s{2,}')

    df = pd.read_csv(flows_csv, low_memory=False, dtype=str)
    df[dur_col] = pd.to_numeric(df[dur_col], errors="coerce")

    req = [ts_col, dur_col, label_col]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns in flows CSV: {miss}")

    # Quick sanity
    sample = df.iloc[0]
    if pkt_window and pkt_window > 0:
        s_ep = _parse_start_ts(sample[ts_col], tz)
        print("Sample ts (raw):", sample[ts_col], file=sys.stderr)
        print(f"Sample packet window: start={pd.to_datetime(s_ep, unit='s', utc=True)} size={pkt_window}", file=sys.stderr)
    else:
        s_ep, e_ep = parse_flow_window(sample[ts_col], df.loc[df.index[0], dur_col],
                                       duration_unit=duration_unit, tz=tz)
        print("Sample ts (raw):", sample[ts_col], file=sys.stderr)
        print("Sample dur (raw):", sample[dur_col], file=sys.stderr)
        print("Sample window (UTC):", pd.to_datetime([s_ep, e_ep], unit="s", utc=True).tolist(), file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["Text","Label"])
        writer.writeheader()

        cur = con.cursor()
        seen = set()
        nrows = len(df) if limit_rows is None else min(limit_rows, len(df))
        for i in range(nrows):
            row = df.iloc[i]

            if pkt_window and pkt_window > 0:
                try:
                    start_ts = _parse_start_ts(str(row[ts_col]), tz)
                except Exception:
                    continue
                cur.execute(
                    f"SELECT {SELECT_COL} FROM packets "
                    "WHERE ts >= ? AND ip_version IS NOT NULL "
                    "ORDER BY ts ASC LIMIT ?",
                    (start_ts, pkt_window)
                )
            else:
                try:
                    start_ts, end_ts = parse_flow_window(str(row[ts_col]), row[dur_col],
                                                         duration_unit=duration_unit, tz=tz)
                except Exception:
                    continue
                cur.execute(
                    f"SELECT {SELECT_COL} FROM packets "
                    "WHERE ts >= ? AND ts <= ? AND ip_version IS NOT NULL "
                    "ORDER BY ts ASC",
                    (start_ts, end_ts)
                )

            # Fetch raw lines and strip the "bad" fields
            headers = []
            for (h,) in cur.fetchall():
                if not h:
                    continue
                # remove key=value tokens we don't want
                s = BAD_FIELDS_RE.sub('', h)
                # collapse multiple spaces left by removals
                s = MULTISPACE_RE.sub(' ', s).strip()
                headers.append(s)

            window_text = "\n".join(headers) if headers else ""

            if dedup_identical_windows and window_text:
                h = hashlib.sha256(window_text.encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)

            writer.writerow({"Text": window_text, "Label": row[label_col]})

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Index PCAP headers into SQLite and build CICFlowMeter-aligned CSV windows.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", help="Path to a single PCAP/PCAPNG")
    src.add_argument("--pcap-dir", help="Path to a directory of PCAP/PCAPNG files (recursive)")

    ap.add_argument("--flows", required=True, help="CICFlowMeter CSV")
    ap.add_argument("--out", required=True, help="Output CSV with Text,Label")
    ap.add_argument("--db", default=None, help="SQLite DB path (defaults near the pcap or dir)")
    ap.add_argument("--ts-col", default="Timestamp")
    ap.add_argument("--dur-col", default="Flow Duration")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--duration-unit", choices=["us","ms","s"], default="us")
    ap.add_argument("--tz", default="America/Moncton", help="Timezone of CSV timestamps (e.g., Atlantic for CICIDS2018)")
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--dedup-windows", action="store_true")
    ap.add_argument("--pkt-window", type=int, default=40, help="If > 0, build each example from the next N packet headers starting at the row's timestamp (overrides duration-based windows).")
    args = ap.parse_args()

    # Choose DB path
    if args.db:
        db_path = args.db
    else:
        base = args.pcap if args.pcap else args.pcap_dir.rstrip(os.sep)
        db_path = os.path.splitext(os.path.abspath(base))[0] + ".packets.sqlite"

    con = ensure_db(db_path)

    if not args.skip_index:
        if args.pcap:
            print(f"[+] Indexing PCAP into {db_path} ...", file=sys.stderr)
            inserted = index_one_pcap(args.pcap, con)
        else:
            print(f"[+] Indexing PCAPs from {args.pcap_dir} into {db_path} ...", file=sys.stderr)
            inserted = index_dir(args.pcap_dir, con)
        print(f"[+] Inserted {inserted} packet headers.", file=sys.stderr)
    else:
        print(f"[+] Skipping index; using existing DB {db_path}", file=sys.stderr)

    print_db_time_range(con, label="SQLite")

    print(f"[+] Building dataset to {args.out} ...", file=sys.stderr)
    build_dataset_from_windows(
        con, args.flows, args.out,
        ts_col=args.ts_col, dur_col=args.dur_col, label_col=args.label_col,
        duration_unit=args.duration_unit, tz=args.tz,
        limit_rows=args.limit_rows, dedup_identical_windows=args.dedup_windows,
        pkt_window=args.pkt_window
    )
    print("[+] Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
