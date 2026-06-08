"""
clean_micro_check.py

Read Microstrain 3DM CSV exports, convert gyro from rad/s to deg/s,
parse the textual `Time` column, and write a cleaned CSV with
`time_s`, `time_ms`, `gyro_*_deg_s`, and `accel_*_g` columns.

Usage examples:
  python clean_micro_check.py --input MICRO_TEST_6.csv
  python clean_micro_check.py --input-dir . --process-all --output-dir cleaned
"""

from __future__ import annotations
import argparse
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd


def newest_file(data_dir: str, pattern: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)), key=os.path.getmtime)
    return files[-1] if files else None


def find_column(cols, keywords):
    for c in cols:
        lc = c.lower()
        for k in keywords:
            if k in lc:
                return c
    return None


def load_microstrain_file(path: str, fallback_samplerate: bool = False):
    """Load a microstrain CSV produced by SensorConnect / Microstrain export.
    Supports files with a 'DATA_START' marker or plain CSV exports with a
    header row. Returns (source_name, t_s, gx_deg, gy_deg, gz_deg, ax, ay, az).
    """
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    # Try DATA_START style first (SensorConnect export)
    ds_idx = None
    for i, L in enumerate(lines):
        if "DATA_START" in L:
            ds_idx = i
            break

    if ds_idx is not None:
        hdr = lines[ds_idx + 1].strip().split(",")
        rows = [line.strip().split(",") for line in lines[ds_idx + 2:] if line.strip()]
        df = pd.DataFrame(rows, columns=hdr)
    else:
        # Fallback: try reading with pandas directly
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse Microstrain file {path}: {e}")

    # Strip device-ID prefix from column names (e.g. "inertial-6288.213393:scaledAccelX" -> "scaledAccelX")
    df.columns = [c.split(":")[-1] if ":" in c else c for c in df.columns]

    # Map columns to canonical names
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "accelx" in lc or "accel_x" in lc or ("accel" in lc and "x" in lc and "y" not in lc and "z" not in lc):
            col_map[c] = "ax"
        elif "accely" in lc or "accel_y" in lc or ("accel" in lc and "y" in lc):
            col_map[c] = "ay"
        elif "accelz" in lc or "accel_z" in lc or ("accel" in lc and "z" in lc):
            col_map[c] = "az"
        elif "gyrox" in lc or "gyro_x" in lc or "rate_x" in lc:
            col_map[c] = "gx"
        elif "gyroy" in lc or "gyro_y" in lc or "rate_y" in lc:
            col_map[c] = "gy"
        elif "gyroz" in lc or "gyro_z" in lc or "rate_z" in lc:
            col_map[c] = "gz"
        elif "referencetime" in lc or "reference_time" in lc:
            col_map[c] = "sensor_time_ns"

    df = df.rename(columns=col_map)

    for c in ["ax", "ay", "az", "gx", "gy", "gz"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    # Microstrain gyro is exported as rad/s; convert to deg/s.
    gx = np.degrees(df["gx"].values.astype(float))
    gy = np.degrees(df["gy"].values.astype(float))
    gz = np.degrees(df["gz"].values.astype(float))

    ax = df["ax"].values.astype(float)
    ay = df["ay"].values.astype(float)
    az = df["az"].values.astype(float)

    # Use internal sensor reference time (nanoseconds, monotonic hardware clock).
    # The wall-clock "Time" column (computer time) is intentionally ignored.
    t = None
    if "sensor_time_ns" in df.columns:
        ref_ns = pd.to_numeric(df["sensor_time_ns"], errors="coerce").values.astype(float)
        t = (ref_ns - ref_ns[0]) / 1e9  # relative seconds from first sample
        print(f"  Using internal sensor referenceTime: {len(t):,} samples, "
              f"duration={t[-1]:.3f}s, mean_dt={float(np.nanmean(np.diff(t)))*1000:.3f}ms")
    elif fallback_samplerate:
        t = np.arange(len(df), dtype=float) / 1000.0
        print("  WARNING: no sensor referenceTime column found; using 1 kHz index fallback")
    else:
        t = np.full(len(df), np.nan, dtype=float)
        print("  WARNING: no sensor referenceTime column found; time_s will be NaN. "
              "Pass --fallback-samplerate to use 1 kHz index instead.")

    source = os.path.basename(path)
    return source, t, gx, gy, gz, ax, ay, az


def build_clean_df_from_micro(path: str, fallback_samplerate: bool = False) -> pd.DataFrame:
    src, t, gx, gy, gz, ax, ay, az = load_microstrain_file(path, fallback_samplerate=fallback_samplerate)
    out = pd.DataFrame(
        {
            "time_s": t,
            "time_ms": t * 1000.0,
            "gyro_x_deg_s": gx,
            "gyro_y_deg_s": gy,
            "gyro_z_deg_s": gz,
            "accel_x_g": ax,
            "accel_y_g": ay,
            "accel_z_g": az,
        }
    )

    # Attach raw values for reference
    out["GX_raw"] = gx
    out["GY_raw"] = gy
    out["GZ_raw"] = gz
    out["AX_raw"] = ax
    out["AY_raw"] = ay
    out["AZ_raw"] = az

    # Wall-clock Time column is intentionally NOT included in cleaned output.
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Input Microstrain CSV file")
    parser.add_argument("--input-dir", "-d", help="Directory to search for Microstrain CSVs")
    parser.add_argument("--process-all", action="store_true", help="Process all matching files in the directory")
    parser.add_argument("--interactive", action="store_true", help="Interactive pick from directory")
    parser.add_argument("--output", "-o", help="Output cleaned CSV file (single file mode)")
    parser.add_argument("--output-dir", help="Directory to write cleaned files", default=None)
    parser.add_argument("--fallback-samplerate", action="store_true", help="If set, use 1 kHz sample-rate fallback when Time parsing fails (default: do not fallback)")
    parser.add_argument("--require-sensor-time", action="store_true", help="Fail if no parsed sensor Time column is present (do not accept host/computer timestamps)")
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Gather candidates
    if args.input:
        files = [os.path.abspath(args.input) if not os.path.isabs(args.input) else args.input]
    else:
        search_dir = args.input_dir if args.input_dir else data_dir
        if not os.path.isabs(search_dir):
            search_dir = os.path.join(data_dir, search_dir)
        patterns = ["MICRO_IMPULSE_2*.csv", "IMUMICRO.csv", "MicroStrain.csv", "IMUMICRO*.csv", "MICRO_TEST_6*.csv", "MICROIMPULSE*.csv", "IMUDATA_*.csv", "SensorConnectData.csv", "IMUReferenceTimeStamp*.csv", "IMU*Reference*.csv"]
        files = []
        for p in patterns:
            files.extend([f for f in glob.glob(os.path.join(search_dir, p)) if not f.endswith("_clean.csv")])
        files = sorted(set(files), key=os.path.getmtime)

    if not files:
        print("No Microstrain files found")
        return

    # Interactive selection
    if args.interactive and not args.process_all and len(files) > 1:
        print("Select a file:")
        for i, f in enumerate(files, start=1):
            print(f" {i:2d}. {os.path.basename(f)}")
        sel = input("Enter number (default 1): ").strip()
        if not sel:
            files = [files[0]]
        else:
            idx = int(sel) - 1
            files = [files[idx]]

    out_dir = args.output_dir
    if out_dir and not os.path.isabs(out_dir):
        out_dir = os.path.join(data_dir, out_dir)

    for path in files:
        print(f"Processing: {os.path.basename(path)}")
        cleaned = build_clean_df_from_micro(path, fallback_samplerate=args.fallback_samplerate)
        # Validate sensor time presence if requested
        if args.require_sensor_time:
            t = cleaned["time_s"].values
            # require at least one finite timestamp and not all NaN
            if not np.isfinite(t).any():
                raise RuntimeError(f"No valid sensor Time parsed for {os.path.basename(path)} — aborting (require-sensor-time set)")
            # require monotonic increasing times where possible
            finite_idx = np.where(np.isfinite(t))[0]
            if len(finite_idx) > 1:
                t_f = t[finite_idx]
                if not np.all(np.diff(t_f) >= -1e-9):
                    raise RuntimeError(f"Parsed sensor Time for {os.path.basename(path)} is not monotonic — aborting (require-sensor-time set)")
        if args.output and len(files) == 1:
            out_path = args.output if os.path.isabs(args.output) else os.path.join(data_dir, args.output)
        else:
            od = out_dir if out_dir else os.path.dirname(path)
            os.makedirs(od, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(od, base + "_clean.csv")
        cleaned.to_csv(out_path, index=False, float_format="%.6f")
        print(f"Saved: {out_path} ({len(cleaned):,} rows)")


if __name__ == "__main__":
    main()
