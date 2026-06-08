"""
plot_cleaned_imu.py

Read the cleaned IMU CSV (IMUData_AnalogCheck_clean.csv) and plot
- top: gyro X/Y/Z (deg/s)
- bottom: accel X/Y/Z (g)

Saves a PNG by default. Use --show to display interactively.
"""
from __future__ import annotations
import argparse
import os
import sys

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot cleaned IMU CSV (gyro deg/s, accel g)")
    parser.add_argument("-i", "--input", default="IMUData_AnalogCheck_clean.csv", help="Input cleaned CSV")
    parser.add_argument("-o", "--output", default="IMUData_AnalogCheck_plot.png", help="Output PNG file")
    parser.add_argument("--time-ms", action="store_true", help="Use time_ms column for x-axis (default: time_s)")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively (uses TkAgg)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(script_dir, args.input)
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    # Choose backend before importing pyplot
    import matplotlib
    if args.show:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(input_path)
    time_col = "time_ms" if args.time_ms else "time_s"
    if time_col not in df.columns:
        print(f"Time column '{time_col}' not found in CSV. Available: {', '.join(df.columns)}", file=sys.stderr)
        sys.exit(2)

    t = df[time_col].values
    x_label = "Time (ms)" if args.time_ms else "Time (s)"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Gyro (deg/s)
    gyro_cols = ["gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s"]
    gyro_labels = ["Gyro X", "Gyro Y", "Gyro Z"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    plotted = False
    for col, lbl, c in zip(gyro_cols, gyro_labels, colors):
        if col in df.columns:
            axes[0].plot(t, df[col].values, label=lbl, color=c, lw=1.2)
            plotted = True
    if not plotted:
        print("No gyro columns found in CSV.")
    axes[0].set_ylabel("Gyro (deg/s)")
    axes[0].grid(True, alpha=0.6)
    axes[0].legend(loc="upper right")

    # Accel (g)
    acc_cols = ["accel_x_g", "accel_y_g", "accel_z_g"]
    acc_labels = ["Accel X", "Accel Y", "Accel Z"]
    plotted = False
    for col, lbl, c in zip(acc_cols, acc_labels, colors):
        if col in df.columns:
            axes[1].plot(t, df[col].values, label=lbl, color=c, lw=1.2)
            plotted = True
    if not plotted:
        print("No accel columns found in CSV.")
    axes[1].set_ylabel("Accel (g)")
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()

    out_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
