"""
align_impulses.py  --  PPS-sync verification via 3-snap protocol

All 3 IMUs are PPS-synced (same clock rate). The only offset is from
when each SDK was loaded (which second boundary the recording started on).

Experiment protocol:
  1. All 3 IMUs recording simultaneously on rigid body (different faces).
  2. Do 3 sharp snaps/strikes separated by at least 1-2 seconds.
  3. The 3 biggest magnitude spikes in each sensor = the 3 snaps.

Alignment logic:
  - SNAP #1  ->  computes the constant SDK-start offset for each sensor vs ADI
  - SNAPS #2 and #3  ->  verification: apply snap#1's shift, check residuals
  - If snaps #2 and #3 land within 1-3 ms  ->  PPS sync confirmed
  - XYZ are NOT compared (sensors on different faces of rigid body)

Usage:
  python align_impulses.py                     # auto-discovers newest combined CSV
  python align_impulses.py --input my.csv
  python align_impulses.py --threshold 2.5     # lower = detect weaker snaps
  python align_impulses.py --min-sep-s 1.5     # increase if two snaps merge
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
REQUIRED_COLS = [
    "sensor", "time_s",
    "gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s",
    "accel_x_g",   "accel_y_g",   "accel_z_g",
]
SENSOR_COLORS = {"ADI": "#e63946", "KERNEL": "#457b9d", "MICRO": "#2a9d8f"}
MATCH_WINDOW_S = 1.0   # seconds -- max gap to accept as a matched impulse pair


# ── helpers ─────────────────────────────────────────────────────────────────

def find_latest_combined(folder):
    candidates = []
    for root, dirs, files in os.walk(folder):
        # skip the alignment output subfolder
        dirs[:] = [d for d in dirs if d.lower() != "alignment"]
        for f in files:
            if f.endswith(".csv") and "combined" in f.lower() and "_aligned" not in f:
                candidates.append(os.path.join(root, f))
    if not candidates:
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d.lower() != "alignment"]
            for f in files:
                if f.endswith(".csv") and "_aligned" not in f:
                    p = os.path.join(root, f)
                    try:
                        peek = pd.read_csv(p, nrows=2)
                        if all(c in peek.columns for c in REQUIRED_COLS):
                            candidates.append(p)
                    except Exception:
                        pass
    return max(candidates, key=os.path.getmtime) if candidates else None


def _moving_avg(x, n):
    if n <= 1:
        return x.copy()
    return np.convolve(x, np.ones(n) / n, mode="same")


def _robust_z(x):
    x = x.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        sd = np.nanstd(x)
        return (x - np.nanmean(x)) / sd if sd > 1e-12 else np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


# ── motion magnitude + impulse detection ────────────────────────────────────

def compute_score(df, sensor, smooth_ms=20.0):
    """Return (time_s, smoothed_score) arrays for one sensor."""
    d = (df[df["sensor"] == sensor]
         .sort_values("time_s")
         .drop_duplicates(subset=["time_s"])
         .dropna(subset=["time_s"]))
    t  = d["time_s"].to_numpy(float)
    gx = d["gyro_x_deg_s"].to_numpy(float)
    gy = d["gyro_y_deg_s"].to_numpy(float)
    gz = d["gyro_z_deg_s"].to_numpy(float)
    ax = d["accel_x_g"].to_numpy(float)
    ay = d["accel_y_g"].to_numpy(float)
    az = d["accel_z_g"].to_numpy(float)

    gyro  = np.sqrt(gx**2 + gy**2 + gz**2)
    accel = np.sqrt(ax**2 + ay**2 + az**2)

    diffs = np.diff(t)
    dt = float(np.nanmedian(diffs[diffs > 0])) if np.any(diffs > 0) else 1e-3
    n_base = max(3, int(0.5 / dt))
    baseline = _moving_avg(
        np.array([np.nanmedian(accel[max(0, i - n_base // 2):i + n_base // 2 + 1])
                  for i in range(len(accel))]), 1)
    accel_dev = np.abs(accel - baseline)

    score = np.sqrt(_robust_z(gyro)**2 + _robust_z(accel_dev)**2)
    return t, _moving_avg(score, max(1, int(smooth_ms * 1e-3 / dt)))


def detect_impulses(t, score, threshold, min_sep_s=0.3):
    """Return list of (time_s, peak_score) for each detected impulse."""
    above = score >= threshold
    impulses = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            seg = score[i:j]
            pk  = int(np.argmax(seg))
            pk_t = float(t[i + pk])
            pk_s = float(seg[pk])
            if not impulses or (pk_t - impulses[-1][0]) >= min_sep_s:
                impulses.append((pk_t, pk_s))
            i = j
        else:
            i += 1
    return impulses   # list of (time_s, score)


N_SNAPS = 3   # number of snaps the experiment is designed around


# ── alignment ────────────────────────────────────────────────────────────────

def top_n_snaps(imps, n=N_SNAPS):
    """
    From all detected impulses pick the N largest by score,
    then re-sort them chronologically.
    """
    if len(imps) < n:
        return sorted(imps, key=lambda x: x[0])   # return whatever we have, sorted by time
    top = sorted(imps, key=lambda x: x[1], reverse=True)[:n]
    return sorted(top, key=lambda x: x[0])         # chronological order


def align_sensors(sensor_scores, threshold, min_sep_s=0.3):
    """
    Detect all impulses, pick the top-3 by magnitude, use all 3 to compute
    a mean shift vs ADI.  Residuals of all 3 snaps are the quality metric.

    Returns:
      snaps     : {sensor: [(local_t, score), ...]}  exactly up to N_SNAPS entries
      all_imps  : {sensor: [(local_t, score), ...]}  every detected impulse
      shifts_s  : {sensor: float}
      matches   : list of per-snap dicts
    """
    if "ADI" not in sensor_scores:
        raise ValueError("ADI sensor not in combined CSV -- cannot use as reference.")

    all_imps = {}
    snaps    = {}
    for sensor, (t, score) in sensor_scores.items():
        imps = detect_impulses(t, score, threshold, min_sep_s)
        all_imps[sensor] = imps
        snaps[sensor]    = top_n_snaps(imps)
        n_found = len(snaps[sensor])
        if n_found < N_SNAPS:
            print(f"  WARNING: {sensor} only has {n_found}/{N_SNAPS} snaps detected "
                  f"(threshold={threshold:.1f}). Try --threshold lower or --min-sep-s lower.")

    # -- SNAP #1 only: compute constant SDK-start offset --------------------
    adi_snaps = snaps.get("ADI", [])
    shifts_s  = {"ADI": 0.0}
    for s, s_snaps in snaps.items():
        if s == "ADI" or not s_snaps or not adi_snaps:
            shifts_s[s] = 0.0
            continue
        shifts_s[s] = adi_snaps[0][0] - s_snaps[0][0]

    # -- build per-snap records (snap#1=anchor, #2+3=verification) ----------
    matches = []
    for i, (adi_t, _) in enumerate(adi_snaps):
        for s, s_snaps in snaps.items():
            if s == "ADI" or i >= len(s_snaps):
                continue
            local_t   = s_snaps[i][0]
            aligned_t = local_t + shifts_s[s]
            delta_ms  = (adi_t - aligned_t) * 1000.0
            matches.append({
                "snap_num":       i + 1,
                "role":           "anchor" if i == 0 else "verification",
                "adi_time_s":     round(adi_t, 4),
                "sensor":         s,
                "sensor_local_s": round(local_t, 4),
                "shift_s":        round(shifts_s[s], 6),
                "aligned_time_s": round(aligned_t, 4),
                "delta_ms":       round(delta_ms, 3),
                "abs_delta_ms":   round(abs(delta_ms), 3),
            })

    return snaps, all_imps, shifts_s, matches


# ── terminal report ──────────────────────────────────────────────────────────

def print_report(snaps, all_imps, shifts_s, matches, threshold, threshold_ms=3.0):
    sensors   = sorted(snaps.keys())
    other     = [s for s in sensors if s != "ADI"]
    adi_snaps = snaps.get("ADI", [])
    bar = "=" * 70

    print("\n" + bar)
    print(f"  IMU PPS-SYNC VERIFICATION  (3-snap protocol)")
    print(f"  Reference: ADI   Detection threshold: {threshold:.2f} sigma")
    print(bar)

    # ── snap detection table ──────────────────────────────────────────────
    col_w = 22
    print(f"\n  TOP-3 SNAPS DETECTED  (sensor-local clock)  [score in brackets]")
    hdr = f"  {'Snap':>14s}  {'ADI':>{col_w}s}" + "".join(f"  {s:>{col_w}s}" for s in other)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    labels = ["#1 (anchor)", "#2 (verify)", "#3 (verify)"]
    for i in range(N_SNAPS):
        adi_str = (f"{adi_snaps[i][0]:.4f} s  [{adi_snaps[i][1]:.1f}]"
                   if i < len(adi_snaps) else "--- not found ---")
        row = f"  {labels[i]:>14s}  {adi_str:>{col_w}s}"
        for s in other:
            s_snaps = snaps.get(s, [])
            s_str = (f"{s_snaps[i][0]:.4f} s  [{s_snaps[i][1]:.1f}]"
                     if i < len(s_snaps) else "--- not found ---")
            row += f"  {s_str:>{col_w}s}"
        print(row)
    print()
    for s in sensors:
        print(f"    {s}: top {len(snaps.get(s,[]))} of "
              f"{len(all_imps.get(s,[]))} total detected impulses")

    # ── SDK-start offset from snap #1 ─────────────────────────────────────
    print(f"\n  SDK-START OFFSET  (computed from snap #1 only)")
    for s in other:
        sh = shifts_s[s]
        direction = "ADI leads" if sh > 0 else "ADI lags "
        print(f"    {s:>8s}:  {sh:+.4f} s  ({direction} by {abs(sh):.4f} s)")

    # ── verification residuals — snaps #2 and #3 ──────────────────────────
    verify = [m for m in matches if m["role"] == "verification"]
    print(f"\n  PPS-SYNC VERIFICATION  (snaps #2 and #3 after applying snap#1 shift)")
    print(f"  Target: < {threshold_ms:.0f} ms  ==>  PPS sync confirmed")
    print(f"  {'Snap':>14s}  {'ADI t':>9s}" +
          "".join(f"  {'ADI - '+s:>16s}" for s in other))
    print("  " + "-" * (16 + 11 + 18 * len(other)))

    within = {s: 0 for s in other}
    for snap_num in [2, 3]:
        row_data = {m["sensor"]: m for m in verify if m["snap_num"] == snap_num}
        adi_t    = next((m["adi_time_s"] for m in row_data.values()), None)
        lbl      = f"#{'2 (verify)' if snap_num == 2 else '3 (verify)'}"
        row = f"  {lbl:>14s}  {adi_t:.4f} s" if adi_t else f"  {lbl:>14s}  ---"
        for s in other:
            m = row_data.get(s)
            if m is None:
                row += f"  {'---':>16s}"
            else:
                r    = m["delta_ms"]
                flag = "  <CHK" if abs(r) > threshold_ms else "  OK  "
                row += f"  {r:>+10.2f} ms{flag}"
                if abs(r) <= threshold_ms:
                    within[s] += 1
        print(row)

    # ── summary ───────────────────────────────────────────────────────────
    n_verify = N_SNAPS - 1
    print(f"\n  RESULT")
    overall = []
    for s in other:
        ok  = within[s]
        pct = ok / n_verify * 100.0
        overall.append(pct)
        rms_v = [m["delta_ms"] for m in verify if m["sensor"] == s]
        rms   = float(np.sqrt(np.mean(np.array(rms_v)**2))) if rms_v else 0.0
        print(f"    {s:>8s}:  {ok}/{n_verify} verification snaps within {threshold_ms:.0f} ms  "
              f"= {pct:.0f}%   RMS = {rms:.2f} ms")

    avg_pct = float(np.mean(overall)) if overall else 0.0
    verdict = ("PPS SYNC CONFIRMED" if avg_pct == 100.0
               else "PARTIAL  -- check flagged snaps" if avg_pct >= 50.0
               else "NOT CONFIRMED  -- re-check data or lower --threshold")

    print(f"\n  OVERALL: {avg_pct:.0f}%  [{verdict}]")
    print(bar + "\n")


# ── plot ─────────────────────────────────────────────────────────────────────

def make_plot(df, sensor_scores, snaps, all_imps, shifts_s, matches, out_png, threshold):
    sensors = sorted(sensor_scores.keys())
    other   = [s for s in sensors if s != "ADI"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("IMU PPS-Sync Verification  (3-snap: #1=anchor, #2+3=verify)",
                 fontsize=13, fontweight="bold")

    # ── before — all impulses faint, top-3 snaps bold with role label ──
    snap_labels = ["anchor", "verify", "verify"]
    for s in sensors:
        t, score = sensor_scores[s]
        col = SENSOR_COLORS.get(s, "#555555")
        ax0.plot(t, score, color=col, lw=1.2, alpha=0.85, label=s)
        for it, sc in all_imps.get(s, []):
            ax0.axvline(it, color=col, ls=":", lw=0.6, alpha=0.35)
        for i, (it, sc) in enumerate(snaps.get(s, [])):
            ax0.axvline(it, color=col, ls="--", lw=1.2, alpha=0.85)
            lbl = snap_labels[i] if i < len(snap_labels) else f"#{i+1}"
            ax0.text(it, sc * 1.06, f"{s}\n{lbl}",
                     fontsize=6.5, color=col, ha="center", va="bottom", fontweight="bold")
    ax0.axhline(threshold, color="#888888", ls=":", lw=0.8, label=f"threshold={threshold:.1f}")
    ax0.set_title("Before alignment — sensor-local clocks  (bold = selected top-3 snaps)")
    ax0.set_xlabel("time_s  (local clock)")
    ax0.set_ylabel("motion magnitude score")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.25)

    # ── after ──
    for s in sensors:
        t, score = sensor_scores[s]
        shift = shifts_s.get(s, 0.0)
        col   = SENSOR_COLORS.get(s, "#555555")
        ax1.plot(t + shift, score, color=col, lw=1.2, alpha=0.85,
                 label=f"{s}  (shift {shift:+.3f} s)")
        for it, _ in snaps.get(s, []):
            ax1.axvline(it + shift, color=col, ls="--", lw=1.2, alpha=0.8)

    # annotate residuals
    adi_snaps = snaps.get("ADI", [])
    for i, (adi_t, _) in enumerate(adi_snaps):
        for j, s in enumerate(other):
            rel = [m for m in matches
                   if m["sensor"] == s and m["snap_num"] == i + 1]
            if rel:
                r   = rel[0]["delta_ms"]
                col = SENSOR_COLORS.get(s, "#555555")
                y_f = 0.90 - j * 0.10 - i * 0.22
                ax1.annotate(
                    f"snap#{i+1} {'anchor' if i==0 else 'verify'}  ADI-{s}: {r:+.2f} ms",
                    xy=(adi_t, y_f), xycoords=("data", "axes fraction"),
                    fontsize=7.5, color=col,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                )

    ax1.set_title("After alignment — shifted to ADI clock")
    ax1.set_xlabel("aligned_time_s  (ADI clock)")
    ax1.set_ylabel("motion magnitude score")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Plot saved:   {out_png}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=(
            "PPS-sync verification via 3-snap protocol.\n"
            "All 3 IMUs on rigid body; do 3 hard snaps/strikes.\n"
            "  Snap #1 -> computes SDK-start offset (shift).\n"
            "  Snaps #2+3 -> verify PPS sync (target: < 3 ms residual).\n\n"
            "Lower --threshold if snaps not detected (default 3.0 sigma).\n"
            "Increase --min-sep-s if consecutive snaps merge (default 0.3 s)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",        "-i", default=None,
                   help="Combined cleaned CSV (auto-discovers newest if omitted)")
    p.add_argument("--out-dir",      "-o", default="clean_data_imus/alignment")
    p.add_argument("--threshold",    "-t", type=float, default=3.0,
                   help="Impulse detection threshold in robust-z sigma (default 3.0)")
    p.add_argument("--smooth-ms",         type=float, default=20.0,
                   help="Smoothing window in ms before detection (default 20)")
    p.add_argument("--min-sep-s",         type=float, default=0.3,
                   help="Minimum separation between impulses in s (default 0.3)")
    p.add_argument("--threshold-ms",      type=float, default=3.0,
                   help="Residual threshold ms for PPS-sync pass/fail (default 3)")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_root = os.path.join(script_dir, "clean_data_imus")

    # ── load CSV ──
    if args.input:
        input_path = (args.input if os.path.isabs(args.input)
                      else os.path.join(script_dir, args.input))
        if not os.path.isfile(input_path):
            print(f"ERROR: file not found: {input_path}")
            sys.exit(1)
    else:
        input_path = find_latest_combined(clean_root)
        if not input_path:
            print("ERROR: No combined clean CSV found. Run clean_all_imu_gui.py first "
                  "or pass --input.")
            sys.exit(1)
        print(f"Using: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    sensors = sorted(df["sensor"].dropna().unique().tolist())
    print(f"Sensors: {sensors}")
    print(f"Computing motion magnitude (threshold={args.threshold:.1f} sigma, "
          f"smooth={args.smooth_ms:.0f} ms)...")

    sensor_scores = {s: compute_score(df, s, smooth_ms=args.smooth_ms)
                     for s in sensors}

    snaps, all_imps, shifts_s, matches = align_sensors(
        sensor_scores, threshold=args.threshold, min_sep_s=args.min_sep_s
    )

    print_report(snaps, all_imps, shifts_s, matches,
                 threshold=args.threshold, threshold_ms=args.threshold_ms)

    # ── apply shifts to CSV ──
    out_df = df.copy()
    out_df["alignment_offset_s"] = out_df["sensor"].map(shifts_s).fillna(0.0)
    out_df["aligned_time_s"]     = out_df["time_s"] + out_df["alignment_offset_s"]
    out_df["aligned_time_ms"]    = out_df["aligned_time_s"] * 1000.0

    # ── save ──
    os.makedirs(args.out_dir, exist_ok=True)
    import re
    stem = os.path.splitext(os.path.basename(input_path))[0]
    stem = re.sub(r'(_aligned)+$', '', stem)   # strip any trailing _aligned suffixes

    aligned_csv = os.path.join(args.out_dir, f"{stem}_aligned.csv")
    report_json = os.path.join(args.out_dir, f"{stem}_alignment_report.json")
    out_png     = os.path.join(args.out_dir, f"{stem}_alignment_plot.png")

    out_df.to_csv(aligned_csv, index=False, float_format="%.6f")
    print(f"Aligned CSV:  {aligned_csv}")

    report = {
        "input": input_path,
        "reference_sensor": "ADI",
        "n_snaps": N_SNAPS,
        "detection_threshold_sigma": args.threshold,
        "smooth_ms": args.smooth_ms,
        "snaps_local_s": {
            s: [[round(t, 6), round(sc, 4)] for t, sc in snaps.get(s, [])]
            for s in sensors
        },
        "all_impulse_counts": {s: len(v) for s, v in all_imps.items()},
        "shifts_applied_s": {s: round(v, 6) for s, v in shifts_s.items()},
        "match_quality": matches,
    }
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report JSON:  {report_json}")

    make_plot(df, sensor_scores, snaps, all_imps, shifts_s, matches, out_png,
              threshold=args.threshold)


if __name__ == "__main__":
    main()
