"""
clean_adi_analog_check.py

Read an ADI "analog check" CSV (e.g. IMUData_AnalogCheck_*.csv),
scale the 32-bit LWR gyro/accel fields into physical units and
write a much cleaner CSV with time (s, ms), gyro (deg/s) and accel (g).

Usage:
  python clean_adi_analog_check.py --input IMUData_AnalogCheck_2026-05-14T13-19-01_0000.csv

If no --input is provided the newest IMUData_AnalogCheck*.csv in the
script directory will be used. The default output is
IMUData_AnalogCheck_clean.csv (same folder).
"""

from __future__ import annotations
import argparse
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd

# Default scaling constants (match your existing plotting script)
GYRO_SCALE_DEFAULT = 10.0
ACCEL_SCALE_DEFAULT = 800.0
LWR_DENOM_BITS = 65536.0

# TIME_STAMP_LWR resolution per the ADIS16577 datasheet:
#   TS_32=0 (default): 16-bit counter width,  resolution = 49.02  μs/LSB
#   TS_32=1          : 32-bit counter width,  resolution = 0.01923 μs/LSB
# NOTE: The counter rollover modulus depends on the sync mode and rate, NOT just the
# bit width.  In scaled/direct external sync modes the counter resets every sync period
# (e.g., 1 Hz sync → modulus ≈ 1 s / 49.02 μs = 20400 LSBs).  The modulus is therefore
# auto-detected from the data so no assumption about sync rate is needed.
TS_RES_DEFAULT_S = 49.02e-6   # seconds per LSB when TS_32=0
TS_RES_TS32_S   = 0.01923e-6  # seconds per LSB when TS_32=1


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


def safe_int64(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int64)


def unwrap_counter(values: np.ndarray, modulus: int) -> np.ndarray:
    """Unwrap a hardware counter given its known rollover modulus."""
    if len(values) == 0:
        return np.zeros(0, dtype=np.int64)
    unwrap = np.zeros(len(values), dtype=np.int64)
    unwrap[0] = values[0]
    half = modulus // 2
    for i in range(1, len(values)):
        d = int(values[i]) - int(values[i - 1])
        if d < -half:
            d += modulus
        unwrap[i] = unwrap[i - 1] + d
    return unwrap


def detect_ts_modulus(ts: np.ndarray) -> int:
    """
    Auto-detect the rollover modulus of TIME_STAMP_LWR from the data.
    Works for any sync mode / sync rate: finds the typical forward step and
    the observed maximum, then computes modulus = round(max / step) * step.
    """
    diffs = np.diff(ts.astype(np.int64))
    fwd = diffs[diffs > 0]
    if len(fwd) == 0:
        raise ValueError("Cannot detect TIME_STAMP_LWR step — no positive increments found")
    step = int(np.median(fwd))
    if step == 0:
        raise ValueError("Median forward step is 0 — TIME_STAMP_LWR appears constant")
    modulus = int(round(int(ts.max()) / step)) * step
    if modulus <= 0:
        raise ValueError(f"Detected counter modulus is non-positive: {modulus}")
    print(f"  TIME_STAMP_LWR: detected step={step} LSB, max={int(ts.max())} LSB, "
          f"modulus={modulus} LSB")
    return modulus


def compute_time_from_timestamp_lwr(df, ts_col: str, ts32: bool = False) -> np.ndarray:
    """
    Convert TIME_STAMP_LWR to seconds using the hardware resolution from the datasheet.
    TS_32=0 (default): 49.02 μs/LSB
    TS_32=1           : 0.01923 μs/LSB
    The rollover modulus is auto-detected from the data so it works correctly in
    both free-running (internal sync) and periodic-reset (external sync) modes.
    """
    ts = safe_int64(df[ts_col]).values.astype(np.int64)
    resolution_s = TS_RES_TS32_S if ts32 else TS_RES_DEFAULT_S
    modulus = detect_ts_modulus(ts)
    unwrap = unwrap_counter(ts, modulus)
    t = (unwrap - unwrap[0]) * resolution_s
    return t.astype(float)


def build_clean_df(df: pd.DataFrame, gyro_scale: float, accel_scale: float, ts32: bool = False) -> pd.DataFrame:
    cols = list(df.columns)
    gx_col = find_column(cols, ["x_gyro_lwr", "x_gyro", "rate_x", "gyrox"])
    gy_col = find_column(cols, ["y_gyro_lwr", "y_gyro", "rate_y", "gyroy"])
    gz_col = find_column(cols, ["z_gyro_lwr", "z_gyro", "rate_z", "gyroz"])
    ax_col = find_column(cols, ["x_accl_lwr", "x_accl", "accelx", "acc_x", "acc1_x"])
    ay_col = find_column(cols, ["y_accl_lwr", "y_accl", "accely", "acc_y", "acc1_y"])
    az_col = find_column(cols, ["z_accl_lwr", "z_accl", "accelz", "acc_z", "acc1_z"])
    ts_col = find_column(cols, ["time_stamp_lwr", "timestamp_lwr"]) 

    if not (gx_col and gy_col and gz_col and ax_col and ay_col and az_col):
        raise ValueError("Required gyro/accel LWR columns not found in CSV. Found columns: " + ",".join(cols))

    # Convert raw LWR values to integers safely
    raw_gx = safe_int64(df[gx_col]).values.astype(np.int64)
    raw_gy = safe_int64(df[gy_col]).values.astype(np.int64)
    raw_gz = safe_int64(df[gz_col]).values.astype(np.int64)
    raw_ax = safe_int64(df[ax_col]).values.astype(np.int64)
    raw_ay = safe_int64(df[ay_col]).values.astype(np.int64)
    raw_az = safe_int64(df[az_col]).values.astype(np.int64)

    # Scale to physical units
    gx = raw_gx.astype(float) / (gyro_scale * LWR_DENOM_BITS)
    gy = raw_gy.astype(float) / (gyro_scale * LWR_DENOM_BITS)
    gz = raw_gz.astype(float) / (gyro_scale * LWR_DENOM_BITS)
    ax = raw_ax.astype(float) / (accel_scale * LWR_DENOM_BITS)
    ay = raw_ay.astype(float) / (accel_scale * LWR_DENOM_BITS)
    az = raw_az.astype(float) / (accel_scale * LWR_DENOM_BITS)

    # Time (require TIME_STAMP_LWR)
    if ts_col and ts_col in df.columns:
        t_s = compute_time_from_timestamp_lwr(df, ts_col, ts32=ts32)
    else:
        raise ValueError("TIME_STAMP_LWR column not found in CSV; logs must include TIME_STAMP_LWR")

    out = pd.DataFrame(
        {
            "time_s": t_s,
            "time_ms": t_s * 1000.0,
            "gyro_x_deg_s": gx,
            "gyro_y_deg_s": gy,
            "gyro_z_deg_s": gz,
            "accel_x_g": ax,
            "accel_y_g": ay,
            "accel_z_g": az,
        }
    )

    # Attach original raw fields for reference
    out["X_GYRO_LWR_raw"] = raw_gx
    out["Y_GYRO_LWR_raw"] = raw_gy
    out["Z_GYRO_LWR_raw"] = raw_gz
    out["X_ACCL_LWR_raw"] = raw_ax
    out["Y_ACCL_LWR_raw"] = raw_ay
    out["Z_ACCL_LWR_raw"] = raw_az

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Input CSV file (IMUData_AnalogCheck_*.csv). If provided, processes this single file.")
    parser.add_argument("--input-dir", "-d", help="Directory containing IMUData_AnalogCheck CSVs to choose from")
    parser.add_argument("--process-all", action="store_true", help="Process all matching CSVs in --input-dir")
    parser.add_argument("--interactive", action="store_true", help="Prompt to select file(s) from --input-dir")
    parser.add_argument("--output", "-o", help="Output cleaned CSV file (single-file mode). If omitted, writes <input_basename>_clean.csv next to the input.", default=None)
    parser.add_argument("--output-dir", help="Directory to write cleaned files (default: same directory as each input)", default=None)
    parser.add_argument("--gyro-scale", type=float, default=GYRO_SCALE_DEFAULT, help="Gyro scale constant")
    parser.add_argument("--accel-scale", type=float, default=ACCEL_SCALE_DEFAULT, help="Accel scale constant")
    parser.add_argument("--ts32", action="store_true",
                        help="Set if TS_32 bit was enabled in MSC_CTRL: uses 32-bit counter at 0.01923 μs/LSB (default: 16-bit at 49.02 μs/LSB)")
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine candidate files
    if args.input:
        input_path = os.path.join(data_dir, args.input) if not os.path.isabs(args.input) else args.input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        selected_files = [input_path]
    else:
        search_dir = args.input_dir if args.input_dir else data_dir
        # resolve relative input-dir against script folder
        if args.input_dir and not os.path.isabs(args.input_dir):
            search_dir = os.path.join(data_dir, args.input_dir)

        candidates = []
        for pat in ("IMUData_AnalogCheck*.csv", "ADI_*.csv"):
            for f in glob.glob(os.path.join(search_dir, pat)):
                if f.endswith("_clean.csv"):
                    continue
                candidates.append(f)
        candidates = sorted(set(candidates), key=os.path.getmtime)
        if not candidates:
            raise FileNotFoundError(f"No raw IMUData_AnalogCheck or ADI CSV found in folder: {search_dir}")

        if args.process_all:
            selected_files = candidates
        elif args.interactive and len(candidates) > 0:
            # Interactive paged selection (drop-down style) showing 1..10 per page.
            # Default interactive folder is an 'analog' subfolder when input_dir not provided.
            if not args.input_dir:
                analog_dir = os.path.join(data_dir, "analog")
                if os.path.isdir(analog_dir):
                    # prefer files in the analog folder if present
                    cand_dir = analog_dir
                    candidates = [f for f in candidates if os.path.dirname(f) == os.path.abspath(cand_dir)] or candidates

            page_size = 10
            page = 0
            selected_files = []
            while True:
                start = page * page_size
                end = min(start + page_size, len(candidates))
                print(f"\nFiles {start+1}..{end} of {len(candidates)}:")
                for i, f in enumerate(candidates[start:end], start=1):
                    print(f"  {i:2d}. {os.path.basename(f)}")
                cmds = []
                if end < len(candidates):
                    cmds.append("n=next")
                if page > 0:
                    cmds.append("p=prev")
                cmds.append("a=all")
                prompt = f"Select 1-{end-start} or {'/'.join(cmds)} (default 1): "
                sel = input(prompt).strip().lower()
                if sel == "" or sel == "1":
                    selected_files = [candidates[start + 0]]
                    break
                if sel == "a":
                    selected_files = candidates
                    break
                if sel == "n":
                    if end < len(candidates):
                        page += 1
                        continue
                    else:
                        print("Already at last page")
                        continue
                if sel == "p":
                    if page > 0:
                        page -= 1
                        continue
                    else:
                        print("Already at first page")
                        continue
                try:
                    idx = int(sel)
                except ValueError:
                    print("Invalid input — enter a number or command")
                    continue
                if idx < 1 or idx > (end - start):
                    print("Selection out of range for this page")
                    continue
                selected_files = [candidates[start + (idx - 1)]]
                break
        else:
            selected_files = [candidates[-1]]

    # Process selected files (single or batch)
    if len(selected_files) > 1 and args.output:
        print("Warning: --output ignored when processing multiple files; writing per-file outputs")

    print(f"Processing {len(selected_files)} file(s)")
    for input_path in selected_files:
        print(f"Loading: {os.path.basename(input_path)}")
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Rows read: {len(df):,}")

        print(f"Timestamp mode: {'TS_32=1 (0.01923 μs/LSB, 32-bit)' if args.ts32 else 'TS_32=0 default (49.02 μs/LSB, 16-bit)'}")
        cleaned = build_clean_df(df, gyro_scale=args.gyro_scale, accel_scale=args.accel_scale, ts32=args.ts32)

        # Determine output path
        if len(selected_files) == 1 and args.output:
            out_path = args.output if os.path.isabs(args.output) else os.path.join(data_dir, args.output)
        else:
            out_dir = args.output_dir if args.output_dir else os.path.dirname(input_path)
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(data_dir, out_dir)
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(input_path))[0]
            out_path = os.path.join(out_dir, base + "_clean.csv")

        cleaned.to_csv(out_path, index=False, float_format="%.6f")
        print(f"Saved cleaned CSV: {out_path} ({len(cleaned):,} rows)")

if __name__ == "__main__":
    main()
