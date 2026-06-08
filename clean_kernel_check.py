"""
clean_kernel_check.py

Read KERNEL IMU text exports (including OrientationTest format),
extract gyro and accel channels, build zero-referenced sensor time,
and write a cleaned CSV matching the shared schema used by other cleaners.

Usage examples:
  python clean_kernel_check.py --input KERNELTEST6.txt
  python clean_kernel_check.py --input OrientationTest.txt --output-dir cleaned
  python clean_kernel_check.py --input-dir . --process-all --output-dir cleaned
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Optional

import numpy as np
import pandas as pd


def newest_file(data_dir: str, pattern: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)), key=os.path.getmtime)
    return files[-1] if files else None


def parse_measurement_rate(lines: list[str]) -> Optional[float]:
    for line in lines:
        if "measurement rate" in line.lower():
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", line.strip())
            if m:
                try:
                    rate = float(m.group(1))
                    if rate > 0:
                        return rate
                except ValueError:
                    pass
    return None


def find_header_index(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        ll = line.lower()
        if "rate_x" in ll and "rate_y" in ll and "rate_z" in ll:
            return i
    raise RuntimeError("Could not find KERNEL data header row (Rate_X/Rate_Y/Rate_Z)")


def build_time_from_second_fraction(ns_values: np.ndarray) -> np.ndarray:
    if len(ns_values) == 0:
        return np.zeros(0, dtype=float)

    # Kernel SecondFraction is nanoseconds within a second, so unwrap at 1e9.
    t = np.zeros(len(ns_values), dtype=float)
    for i in range(1, len(ns_values)):
        dt = float(ns_values[i]) - float(ns_values[i - 1])
        while dt < 0:
            dt += 1e9
        t[i] = t[i - 1] + dt / 1e9
    return t


def load_kernel_file(path: str):
    """Return (source_name, t_s, gx, gy, gz, ax, ay, az)."""
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    rate_hz = parse_measurement_rate(lines)
    hdr_idx = find_header_index(lines)
    headers = lines[hdr_idx].strip().split()
    h_idx = {h: i for i, h in enumerate(headers)}

    # Required gyro columns
    required = ["Rate_X", "Rate_Y", "Rate_Z"]
    for c in required:
        if c not in h_idx:
            raise RuntimeError(f"Missing required column: {c}")

    # Accel columns vary by export profile.
    if all(k in h_idx for k in ["Acc1_X", "Acc1_Y", "Acc1_Z"]):
        ax_name, ay_name, az_name = "Acc1_X", "Acc1_Y", "Acc1_Z"
    elif all(k in h_idx for k in ["Acc_X", "Acc_Y", "Acc_Z"]):
        ax_name, ay_name, az_name = "Acc_X", "Acc_Y", "Acc_Z"
    elif all(k in h_idx for k in ["Acc2_X", "Acc2_Y", "Acc2_Z"]):
        ax_name, ay_name, az_name = "Acc2_X", "Acc2_Y", "Acc2_Z"
    else:
        raise RuntimeError("No supported accel column triplet found (Acc1_*, Acc_*, or Acc2_*)")

    rows = []
    for line in lines[hdr_idx + 1 :]:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        need_idx = max(
            h_idx["Rate_X"],
            h_idx["Rate_Y"],
            h_idx["Rate_Z"],
            h_idx[ax_name],
            h_idx[ay_name],
            h_idx[az_name],
            h_idx.get("SecondFraction", 0),
        )
        if len(parts) <= need_idx:
            continue

        try:
            gx = float(parts[h_idx["Rate_X"]])
            gy = float(parts[h_idx["Rate_Y"]])
            gz = float(parts[h_idx["Rate_Z"]])
            ax = float(parts[h_idx[ax_name]])
            ay = float(parts[h_idx[ay_name]])
            az = float(parts[h_idx[az_name]])
            second_fraction = None
            if "SecondFraction" in h_idx:
                second_fraction = float(parts[h_idx["SecondFraction"]])
            rows.append((gx, gy, gz, ax, ay, az, second_fraction))
        except ValueError:
            continue

    if not rows:
        raise RuntimeError("No numeric sensor rows were parsed from file")

    arr = np.array(rows, dtype=object)
    gx = arr[:, 0].astype(float)
    gy = arr[:, 1].astype(float)
    gz = arr[:, 2].astype(float)
    ax = arr[:, 3].astype(float)
    ay = arr[:, 4].astype(float)
    az = arr[:, 5].astype(float)

    t = None
    if "SecondFraction" in h_idx:
        ns = arr[:, 6].astype(float)
        t = build_time_from_second_fraction(ns)
        mean_dt_ms = float(np.nanmean(np.diff(t)) * 1000.0) if len(t) > 1 else float("nan")
        print(
            f"  Using SecondFraction sensor time: {len(t):,} samples, "
            f"duration={t[-1]:.3f}s, mean_dt={mean_dt_ms:.3f}ms"
        )
    elif rate_hz is not None:
        t = np.arange(len(gx), dtype=float) / rate_hz
        print(
            f"  No SecondFraction column; using Measurement rate from file header: "
            f"{rate_hz:g} Hz"
        )
    else:
        t = np.arange(len(gx), dtype=float) / 1000.0
        print("  No SecondFraction and no Measurement rate; defaulting to 1 kHz timeline")

    source = os.path.basename(path)
    return source, t, gx, gy, gz, ax, ay, az


def build_clean_df_from_kernel(path: str) -> pd.DataFrame:
    src, t, gx, gy, gz, ax, ay, az = load_kernel_file(path)
    _ = src
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

    out["GX_raw"] = gx
    out["GY_raw"] = gy
    out["GZ_raw"] = gz
    out["AX_raw"] = ax
    out["AY_raw"] = ay
    out["AZ_raw"] = az
    return out


def find_candidates(search_dir: str) -> list[str]:
    patterns = [
        "KERNEL*.txt",
        "*KERNEL*.txt",
        "OrientationTest*.txt",
        "ORIENTATION*.txt",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(search_dir, p)))
    files = [f for f in set(files) if not f.endswith("_clean.csv")]
    return sorted(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Input KERNEL text file")
    parser.add_argument("--input-dir", "-d", help="Directory to search for KERNEL files")
    parser.add_argument("--process-all", action="store_true", help="Process all matching files in the directory")
    parser.add_argument("--interactive", action="store_true", help="Interactive pick from directory")
    parser.add_argument("--output", "-o", help="Output cleaned CSV file (single file mode)")
    parser.add_argument("--output-dir", help="Directory to write cleaned files", default=None)
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))

    if args.input:
        files = [os.path.abspath(args.input) if not os.path.isabs(args.input) else args.input]
    else:
        search_dir = args.input_dir if args.input_dir else data_dir
        if not os.path.isabs(search_dir):
            search_dir = os.path.join(data_dir, search_dir)
        files = find_candidates(search_dir)

    if not files:
        print("No KERNEL files found")
        return

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
        cleaned = build_clean_df_from_kernel(path)
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
