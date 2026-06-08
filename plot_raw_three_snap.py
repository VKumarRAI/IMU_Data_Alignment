import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GYRO_SCALE = 10.0
ACCEL_SCALE = 800.0


def newest(pattern):
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)), key=os.path.getmtime)
    return files[-1] if files else None


def load_adi577():
    burst_path = newest("ADIS1657x_Burst_*.csv")
    impulse2_path = newest(os.path.join("May12", "ADI_IMPULSE2_*.csv"))
    reglog_path = (
        impulse2_path
        or newest("ADIMU1000HzTest_*.csv")
        or newest("ADIMU1000Hz_*.csv")
        or newest("IMUDATATESTSYNC_*.csv")
        or newest("IMU_DATA_TEST_ADI_*.csv")
        or newest("ADI_6_TEST_*.csv")
        or newest("IMUIMPULSE4_*.csv")
        or newest(os.path.join("May12", "newimpulse", "IMUIMPULSE4_*.csv"))
    )

    if burst_path:
        raw = pd.read_csv(burst_path)
        df = raw[raw["BURST_CHECKSUM"] != 0].copy()
        df = df.drop_duplicates(subset=["X_GYRO_LWR", "Y_GYRO_LWR", "Z_GYRO_LWR", "DATA_CNTR"]).reset_index(drop=True)
        gx = df["X_GYRO_LWR"].values.astype(np.int64) / (GYRO_SCALE * 65536.0)
        gy = df["Y_GYRO_LWR"].values.astype(np.int64) / (GYRO_SCALE * 65536.0)
        gz = df["Z_GYRO_LWR"].values.astype(np.int64) / (GYRO_SCALE * 65536.0)
        ax = df["X_ACCL_LWR"].values.astype(np.int64) / (ACCEL_SCALE * 65536.0)
        ay = df["Y_ACCL_LWR"].values.astype(np.int64) / (ACCEL_SCALE * 65536.0)
        az = df["Z_ACCL_LWR"].values.astype(np.int64) / (ACCEL_SCALE * 65536.0)

        cntr = df["DATA_CNTR"].values.astype(np.int64)
        rollover = int(cntr.max()) + 10
        unwrap = np.zeros(len(cntr), dtype=np.int64)
        unwrap[0] = cntr[0]
        for i in range(1, len(cntr)):
            d = cntr[i] - cntr[i - 1]
            if d < -(rollover // 2):
                d += rollover
            unwrap[i] = unwrap[i - 1] + d
        t = (unwrap - unwrap[0]) / 10000.0
        source = os.path.basename(burst_path)
        return source, t, gx, gy, gz, ax, ay, az

    if reglog_path:
        df = pd.read_csv(reglog_path)
        gx = df["X_GYRO_LWR"].values.astype(np.int64) / (GYRO_SCALE * 65536.0)
        gy = df["Y_GYRO_LWR"].values.astype(np.int64) / (GYRO_SCALE * 65536.0)
        gz = df["Z_GYRO_LWR"].values.astype(np.int64) / (GYRO_SCALE * 65536.0)
        ax = df["X_ACCL_LWR"].values.astype(np.int64) / (ACCEL_SCALE * 65536.0)
        ay = df["Y_ACCL_LWR"].values.astype(np.int64) / (ACCEL_SCALE * 65536.0)
        az = df["Z_ACCL_LWR"].values.astype(np.int64) / (ACCEL_SCALE * 65536.0)

        # Raw time from sample index at 1 kHz output rate.
        t = np.arange(len(df), dtype=float) / 1000.0
        source = os.path.basename(reglog_path)
        return source, t, gx, gy, gz, ax, ay, az

    raise FileNotFoundError("No ADI CSV found")


def load_kernel220():
    k26_path = (
        newest(os.path.join("May12", "KERNELIMPULSETEST2*.txt"))
        or newest("K26*.txt")
        or newest("KERNEL_*.txt")
        or newest("KERNELTEST6*.txt")
    )
    if not k26_path:
        raise FileNotFoundError("No Kernel 220 TXT found")

    with open(k26_path, "r") as f:
        lines = f.readlines()

    header_idx = next(i for i, line in enumerate(lines) if "Rate_X" in line)
    headers = lines[header_idx].split()
    rows = []
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) >= len(headers):
            try:
                rows.append([float(x) for x in parts[:len(headers)]])
            except ValueError:
                pass

    df = pd.DataFrame(rows, columns=headers)
    gx = df["Rate_X"].values.astype(float)
    gy = df["Rate_Y"].values.astype(float)
    gz = df["Rate_Z"].values.astype(float)
    ax = df["Acc1_X"].values.astype(float)
    ay = df["Acc1_Y"].values.astype(float)
    az = df["Acc1_Z"].values.astype(float)

    # Raw time from SecondFraction (nanoseconds in each GPS second), unwrapped.
    ns = df["SecondFraction"].values.astype(np.int64)
    t = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        dt = ns[i] - ns[i - 1]
        if dt < 0:
            dt += int(1e9)
        t[i] = t[i - 1] + dt / 1e9

    source = os.path.basename(k26_path)
    return source, t, gx, gy, gz, ax, ay, az


def load_microstrain_3dm():
    ms_path = (
        newest(os.path.join("May12", "MICRO_IMPULSE_2*.csv"))
        or newest("IMUMICRO.csv")
        or newest("MicroStrain.csv")
        or newest("IMUMICRO*.csv")
        or newest("MICRO_TEST_6*.csv")
        or newest("MICROIMPULSE*.csv")
        or newest(os.path.join("May12", "newimpulse", "MICROIMPULSE*.csv"))
        or newest("IMUDATA_*.csv")
        or newest("SensorConnectData.csv")
    )
    if not ms_path:
        raise FileNotFoundError("No Microstrain 3DM CSV found")

    with open(ms_path, "r") as f:
        lines = f.readlines()

    ds_idx = next(i for i, line in enumerate(lines) if "DATA_START" in line)
    hdr = lines[ds_idx + 1].strip().split(",")
    df = pd.DataFrame([line.strip().split(",") for line in lines[ds_idx + 2:] if line.strip()], columns=hdr)

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "accelx" in lc:
            col_map[c] = "ax"
        elif "accely" in lc:
            col_map[c] = "ay"
        elif "accelz" in lc:
            col_map[c] = "az"
        elif "gyrox" in lc:
            col_map[c] = "gx"
        elif "gyroy" in lc:
            col_map[c] = "gy"
        elif "gyroz" in lc:
            col_map[c] = "gz"

    df = df.rename(columns=col_map)
    for c in ["ax", "ay", "az", "gx", "gy", "gz"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Microstrain gyro is rad/s in export; convert to deg/s.
    gx = np.degrees(df["gx"].values.astype(float))
    gy = np.degrees(df["gy"].values.astype(float))
    gz = np.degrees(df["gz"].values.astype(float))
    ax = df["ax"].values.astype(float)
    ay = df["ay"].values.astype(float)
    az = df["az"].values.astype(float)

    ts = df["Time"].str.replace(r"(\.\d{6})\d+", r"\1", regex=True)
    dt = pd.to_datetime(ts, format="%m/%d/%y %H:%M:%S.%f", errors="coerce")
    t = (dt - dt.iloc[0]).dt.total_seconds().values.astype(float)

    source = os.path.basename(ms_path)
    return source, t, gx, gy, gz, ax, ay, az


def main():
    adi_src, adi_t, adi_gx, adi_gy, adi_gz, adi_ax, adi_ay, adi_az = load_adi577()
    k_src, k_t, k_gx, k_gy, k_gz, k_ax, k_ay, k_az = load_kernel220()
    ms_src, ms_t, ms_gx, ms_gy, ms_gz, ms_ax, ms_ay, ms_az = load_microstrain_3dm()

    print(f"ADI577:          {adi_src}  |  {len(adi_t):,} samples  |  {adi_t[-1]*1000.0:.1f} ms")
    print(f"Kernel 220:      {k_src}  |  {len(k_t):,} samples  |  {k_t[-1]*1000.0:.1f} ms")
    print(f"Microstrain 3DM: {ms_src}  |  {len(ms_t):,} samples  |  {ms_t[-1]*1000.0:.1f} ms")

    # Find gyro magnitude and snap to the first major impulse per sensor.
    adi_gmag = np.sqrt(adi_gx**2 + adi_gy**2 + adi_gz**2)
    k_gmag = np.sqrt(k_gx**2 + k_gy**2 + k_gz**2)
    ms_gmag = np.sqrt(ms_gx**2 + ms_gy**2 + ms_gz**2)

    def snap_from_big_impulse(sig):
        # Use dominant global impulse as alignment anchor for snap-start recordings.
        return int(np.argmax(sig))

    def post_snap_impulses(sig, t, snap_idx, start_s=0.02, end_s=1.0, min_spacing_s=0.05, max_events=8):
        # Detect multiple local maxima after the snap event (t=0 after alignment).
        if len(sig) < 3:
            return []

        rel_t = t - t[snap_idx]
        local = np.where((sig[1:-1] > sig[:-2]) & (sig[1:-1] >= sig[2:]))[0] + 1
        local = [int(i) for i in local if start_s <= rel_t[i] <= end_s]
        if not local:
            return []

        local_vals = np.array([sig[i] for i in local], dtype=float)
        snap_amp = float(sig[snap_idx])
        threshold = max(0.2 * snap_amp, float(np.percentile(local_vals, 65)))

        # Pick strongest peaks first, then enforce minimum temporal spacing.
        chosen = []
        for i in sorted(local, key=lambda j: sig[j], reverse=True):
            if sig[i] < threshold:
                continue
            if all(abs(rel_t[i] - rel_t[j]) >= min_spacing_s for j in chosen):
                chosen.append(i)

        chosen = sorted(chosen, key=lambda i: rel_t[i])[:max_events]
        return chosen

    adi_peak_idx = snap_from_big_impulse(adi_gmag)
    k_peak_idx = snap_from_big_impulse(k_gmag)
    ms_peak_idx = snap_from_big_impulse(ms_gmag)

    adi_peak_t = adi_t[adi_peak_idx]
    k_peak_t = k_t[k_peak_idx]
    ms_peak_t = ms_t[ms_peak_idx]

    adi_post_idx = post_snap_impulses(adi_gmag, adi_t, adi_peak_idx)
    k_post_idx = post_snap_impulses(k_gmag, k_t, k_peak_idx)
    ms_post_idx = post_snap_impulses(ms_gmag, ms_t, ms_peak_idx)

    print(f"\n=== MOTION SNAP DETECTION (DOMINANT IMPULSE) ===")
    print(f"ADI577 peak gyro:          {adi_gmag[adi_peak_idx]:.1f} deg/s at sample {adi_peak_idx:,} (t={adi_peak_t*1000.0:.3f} ms)")
    print(f"Kernel 220 peak gyro:      {k_gmag[k_peak_idx]:.1f} deg/s at sample {k_peak_idx:,} (t={k_peak_t*1000.0:.3f} ms)")
    print(f"Microstrain 3DM peak gyro: {ms_gmag[ms_peak_idx]:.1f} deg/s at sample {ms_peak_idx:,} (t={ms_peak_t*1000.0:.3f} ms)")

    print(f"\n=== SNAP TIMING RELATIVE TO EACH SENSOR START ===")
    print(f"ADI577:           {adi_peak_t*1000.0:.3f} ms after start")
    print(f"Kernel 220:       {k_peak_t*1000.0:.3f} ms after start")
    print(f"Microstrain 3DM:  {ms_peak_t*1000.0:.3f} ms after start")
    
    print(f"\n=== TIME OFFSET BETWEEN SNAP EVENTS ===")
    print(f"K220 snap vs ADI snap:     {(k_peak_t - adi_peak_t)*1000:.2f} ms")
    print(f"Microstrain snap vs ADI:   {(ms_peak_t - adi_peak_t)*1000:.2f} ms")
    print(f"Microstrain snap vs K220:  {(ms_peak_t - k_peak_t)*1000:.2f} ms")

    def print_post_snap_events(name, sig, t, snap_idx, event_idx):
        if not event_idx:
            print(f"{name:<16} no post-snap impulses detected in +20ms..+1000ms window")
            return
        rel_ms = [(t[i] - t[snap_idx]) * 1000.0 for i in event_idx]
        amps = [sig[i] for i in event_idx]
        print(f"{name:<16} {len(event_idx)} events after snap:")
        for n, (tt, aa) in enumerate(zip(rel_ms, amps), start=1):
            print(f"  {n:>2}. t={tt:8.2f} ms, gyro|mag|={aa:8.2f} deg/s")

    print(f"\n=== POST-SNAP IMPULSE EVENTS (MULTIPLE) ===")
    print_post_snap_events("ADI577", adi_gmag, adi_t, adi_peak_idx, adi_post_idx)
    print_post_snap_events("Kernel 220", k_gmag, k_t, k_peak_idx, k_post_idx)
    print_post_snap_events("Microstrain", ms_gmag, ms_t, ms_peak_idx, ms_post_idx)

    # Align all to peak snap (t=0 at peak for all three)
    adi_t_aligned = adi_t - adi_peak_t
    k_t_aligned = k_t - k_peak_t
    ms_t_aligned = ms_t - ms_peak_t

    def xcorr_alignment_metrics(t1, s1, t2, s2, win_s=(0.02, 1.0), dt_s=0.001, max_lag_s=0.05):
        # Interpolate both series on a common post-snap grid, then compute normalized cross-correlation.
        w0, w1 = win_s
        t0 = max(float(np.nanmin(t1)), float(np.nanmin(t2)), w0)
        t1_end = min(float(np.nanmax(t1)), float(np.nanmax(t2)), w1)
        if t1_end <= t0 + 5 * dt_s:
            return None

        grid = np.arange(t0, t1_end, dt_s)
        if len(grid) < 16:
            return None

        v1 = np.interp(grid, t1, s1)
        v2 = np.interp(grid, t2, s2)

        sd1 = float(np.std(v1))
        sd2 = float(np.std(v2))
        if sd1 < 1e-9 or sd2 < 1e-9:
            return None

        v1z = (v1 - np.mean(v1)) / sd1
        v2z = (v2 - np.mean(v2)) / sd2

        corr = np.correlate(v1z, v2z, mode="full") / len(v1z)
        lags = np.arange(-len(v1z) + 1, len(v1z)) * dt_s

        lag_mask = np.abs(lags) <= max_lag_s
        if not np.any(lag_mask):
            return None

        corr_limited = corr[lag_mask]
        lags_limited = lags[lag_mask]
        best_i = int(np.argmax(corr_limited))
        best_lag_ms = float(lags_limited[best_i] * 1000.0)
        best_corr = float(corr_limited[best_i])
        zero_i = int(np.argmin(np.abs(lags_limited)))
        zero_corr = float(corr_limited[zero_i])

        return {
            "lag_ms": best_lag_ms,
            "corr_peak": best_corr,
            "corr_zero": zero_corr,
            "samples": int(len(grid)),
            "window_ms": (t0 * 1000.0, t1_end * 1000.0),
        }

    pair_metrics = [
        ("ADI577 vs Kernel 220", xcorr_alignment_metrics(adi_t_aligned, adi_gmag, k_t_aligned, k_gmag)),
        ("ADI577 vs Microstrain", xcorr_alignment_metrics(adi_t_aligned, adi_gmag, ms_t_aligned, ms_gmag)),
        ("Kernel 220 vs Microstrain", xcorr_alignment_metrics(k_t_aligned, k_gmag, ms_t_aligned, ms_gmag)),
    ]

    print("\n=== CROSS-CORRELATION ALIGNMENT (POST-SNAP SERIES) ===")
    for name, m in pair_metrics:
        if m is None:
            print(f"{name:<24} insufficient overlap for xcorr metrics")
            continue
        w0_ms, w1_ms = m["window_ms"]
        print(
            f"{name:<24} lag={m['lag_ms']:7.2f} ms | "
            f"xcorr_peak={m['corr_peak']:.4f} | xcorr_at_0={m['corr_zero']:.4f} | "
            f"win={w0_ms:.0f}..{w1_ms:.0f} ms | n={m['samples']}"
        )

    adi_post_ms = [(adi_t[i] - adi_peak_t) * 1000.0 for i in adi_post_idx]
    k_post_ms = [(k_t[i] - k_peak_t) * 1000.0 for i in k_post_idx]
    ms_post_ms = [(ms_t[i] - ms_peak_t) * 1000.0 for i in ms_post_idx]

    # Keep a broad data window available; the visible x-limits are set separately.
    adi_mask = (adi_t_aligned >= -0.5) & (adi_t_aligned <= 1.2)
    k_mask = (k_t_aligned >= -0.5) & (k_t_aligned <= 1.2)
    ms_mask = (ms_t_aligned >= -0.5) & (ms_t_aligned <= 1.2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle("IMU Data Aligned to Motion Snap (t=0 ms = peak gyro event)", fontsize=14, fontweight='bold')

    colors = {
        "ADI577": "#1f77b4",
        "Kernel 220": "#d62728",
        "Microstrain 3DM": "#2ca02c",
    }

    channels = [
        ("Gyro X (deg/s)", (adi_t_aligned[adi_mask] * 1000.0, adi_gx[adi_mask]), (k_t_aligned[k_mask] * 1000.0, k_gx[k_mask]), (ms_t_aligned[ms_mask] * 1000.0, ms_gx[ms_mask])),
        ("Gyro Y (deg/s)", (adi_t_aligned[adi_mask] * 1000.0, adi_gy[adi_mask]), (k_t_aligned[k_mask] * 1000.0, k_gy[k_mask]), (ms_t_aligned[ms_mask] * 1000.0, ms_gy[ms_mask])),
        ("Gyro Z (deg/s)", (adi_t_aligned[adi_mask] * 1000.0, adi_gz[adi_mask]), (k_t_aligned[k_mask] * 1000.0, k_gz[k_mask]), (ms_t_aligned[ms_mask] * 1000.0, ms_gz[ms_mask])),
        ("Accel X (g)",    (adi_t_aligned[adi_mask] * 1000.0, adi_ax[adi_mask]), (k_t_aligned[k_mask] * 1000.0, k_ax[k_mask]), (ms_t_aligned[ms_mask] * 1000.0, ms_ax[ms_mask])),
        ("Accel Y (g)",    (adi_t_aligned[adi_mask] * 1000.0, adi_ay[adi_mask]), (k_t_aligned[k_mask] * 1000.0, k_ay[k_mask]), (ms_t_aligned[ms_mask] * 1000.0, ms_ay[ms_mask])),
        ("Accel Z (g)",    (adi_t_aligned[adi_mask] * 1000.0, adi_az[adi_mask]), (k_t_aligned[k_mask] * 1000.0, k_az[k_mask]), (ms_t_aligned[ms_mask] * 1000.0, ms_az[ms_mask])),
    ]

    for i, (title, adi_pair, k_pair, ms_pair) in enumerate(channels):
        r = i // 3
        c = i % 3
        ax = axes[r, c]

        ax.axvline(0, color='k', lw=2, ls='--', alpha=0.7, label='Snap point (t=0)')
        for t_ms in adi_post_ms:
            ax.axvline(t_ms, color=colors["ADI577"], lw=0.9, ls=':', alpha=0.25)
        for t_ms in k_post_ms:
            ax.axvline(t_ms, color=colors["Kernel 220"], lw=0.9, ls=':', alpha=0.25)
        for t_ms in ms_post_ms:
            ax.axvline(t_ms, color=colors["Microstrain 3DM"], lw=0.9, ls=':', alpha=0.25)

        ax.plot(adi_pair[0], adi_pair[1], color=colors["ADI577"], lw=1.2, label="ADI577")
        ax.plot(k_pair[0], k_pair[1], color=colors["Kernel 220"], lw=1.2, label="Kernel 220")
        ax.plot(ms_pair[0], ms_pair[1], color=colors["Microstrain 3DM"], lw=1.2, label="Microstrain 3DM")
        
        # Mark impulse peaks with markers
        adi_snap_val = adi_pair[1][np.argmin(np.abs(adi_pair[0] - 0))] if len(adi_pair[0]) > 0 else 0
        k_snap_val = k_pair[1][np.argmin(np.abs(k_pair[0] - 0))] if len(k_pair[0]) > 0 else 0
        ms_snap_val = ms_pair[1][np.argmin(np.abs(ms_pair[0] - 0))] if len(ms_pair[0]) > 0 else 0
        
        ax.plot(0, adi_snap_val, 'o', color=colors["ADI577"], markersize=8, markeredgecolor='black', markeredgewidth=1)
        ax.plot(0, k_snap_val, 's', color=colors["Kernel 220"], markersize=8, markeredgecolor='black', markeredgewidth=1)
        ax.plot(0, ms_snap_val, '^', color=colors["Microstrain 3DM"], markersize=8, markeredgecolor='black', markeredgewidth=1)
        
        # Mark post-snap impulses
        for idx in adi_post_idx:
            t_val = (adi_t[idx] - adi_peak_t) * 1000.0
            if -50 <= t_val <= 150:
                val = adi_gmag[idx] if "Gyro" in title else adi_ax[idx] if "Accel X" in title else adi_ay[idx] if "Accel Y" in title else adi_az[idx]
                ax.plot(t_val, val, 'o', color=colors["ADI577"], markersize=5, alpha=0.6)
        
        for idx in k_post_idx:
            t_val = (k_t[idx] - k_peak_t) * 1000.0
            if -50 <= t_val <= 150:
                val = k_gmag[idx] if "Gyro" in title else k_ax[idx] if "Accel X" in title else k_ay[idx] if "Accel Y" in title else k_az[idx]
                ax.plot(t_val, val, 's', color=colors["Kernel 220"], markersize=5, alpha=0.6)
        
        for idx in ms_post_idx:
            t_val = (ms_t[idx] - ms_peak_t) * 1000.0
            if -50 <= t_val <= 150:
                val = ms_gmag[idx] if "Gyro" in title else ms_ax[idx] if "Accel X" in title else ms_ay[idx] if "Accel Y" in title else ms_az[idx]
                ax.plot(t_val, val, '^', color=colors["Microstrain 3DM"], markersize=5, alpha=0.6)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel("Value", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, axis='x', alpha=0.7, linestyle='-', linewidth=1.2)
        ax.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.5)
        ax.grid(True, axis='x', alpha=0.4, which='minor', linestyle=':', linewidth=0.7)
        ax.grid(True, axis='y', alpha=0.15, which='minor', linestyle=':')
        ax.minorticks_on()

        # Fine x-axis ruler for precise ms alignment checking
        ax.xaxis.set_major_locator(MultipleLocator(10.0))
        ax.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.tick_params(axis='x', which='minor', labelsize=6)

    axes[-1, 0].set_xlabel("Time relative to snap (ms)", fontsize=10, fontweight='bold')
    axes[-1, 1].set_xlabel("Time relative to snap (ms)", fontsize=10, fontweight='bold')
    axes[-1, 2].set_xlabel("Time relative to snap (ms)", fontsize=10, fontweight='bold')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10)

    def autoscale_y_in_window(ax, x_left, x_right):
        y_parts = []
        for line in ax.lines:
            x = np.asarray(line.get_xdata())
            y = np.asarray(line.get_ydata())
            if x.size < 3 or y.size != x.size:
                continue
            m = (x >= x_left) & (x <= x_right) & np.isfinite(y)
            if np.any(m):
                y_parts.append(y[m])
        if not y_parts:
            return
        y_all = np.concatenate(y_parts)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        span = max(y_max - y_min, 1e-6)
        pad = 0.08 * span
        ax.set_ylim(y_min - pad, y_max + pad)

    # Default view: zoom into the main snap peak for details
    default_xlim = (-50.0, 150.0)
    for ax in axes.ravel():
        ax.set_xlim(*default_xlim)
        autoscale_y_in_window(ax, *default_xlim)

    # Add visual ruler lines every 10ms for fast eyeballing of relative peak timing.
    for ax in axes.ravel():
        for x in np.arange(-50.0, 151.0, 10.0):
            ax.axvline(x, color='0.3', lw=0.8, ls='-', alpha=0.3, zorder=0)

    # MATLAB-like cursor for inspection
    flat_axes = axes.ravel()
    cursor_lines = [ax.axvline(np.nan, color="0.25", lw=0.8, ls=":", alpha=0.9) for ax in flat_axes]
    readout = fig.text(
        0.01,
        0.985,
        "Hover=read | Click=lock | Esc=clear | Scroll=zoom | 0:full view, 1:zoom peak",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.9),
    )
    cursor_state = {"locked": False}

    def set_cursor_x(x_ms):
        for line in cursor_lines:
            line.set_xdata([x_ms, x_ms])
            line.set_visible(True)

    def on_move(event):
        if cursor_state["locked"]:
            return
        if event.inaxes not in flat_axes or event.xdata is None or event.ydata is None:
            return
        set_cursor_x(event.xdata)
        readout.set_text(f"t={event.xdata:.1f} ms | {event.inaxes.get_title()}: {event.ydata:.4f}")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes not in flat_axes or event.xdata is None or event.ydata is None:
            return
        cursor_state["locked"] = not cursor_state["locked"]
        set_cursor_x(event.xdata)
        status = "LOCKED" if cursor_state["locked"] else "UNLOCKED"
        readout.set_text(f"[{status}] t={event.xdata:.1f} ms | {event.inaxes.get_title()}: {event.ydata:.4f}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "escape":
            cursor_state["locked"] = False
            for line in cursor_lines:
                line.set_visible(False)
            readout.set_text("Hover=read | Click=lock | Esc=clear | Scroll=zoom | 0:full view, 1:zoom peak")
            fig.canvas.draw_idle()
        elif event.key == "0":
            for ax in flat_axes:
                ax.set_xlim(-100.0, 350.0)
                autoscale_y_in_window(ax, -100.0, 350.0)
            readout.set_text("View: full range -100 to 350 ms")
            fig.canvas.draw_idle()
        elif event.key == "1":
            for ax in flat_axes:
                ax.set_xlim(-50.0, 150.0)
                autoscale_y_in_window(ax, -50.0, 150.0)
            readout.set_text("Zoomed: peak view -50 to 150 ms")
            fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes not in flat_axes or event.xdata is None:
            return
        left, right = event.inaxes.get_xlim()
        span = right - left
        zoom_in = event.button == "up"
        scale = 0.8 if zoom_in else 1.25
        new_span = span * scale
        center = event.xdata
        new_left = center - 0.5 * new_span
        new_right = center + 0.5 * new_span
        for ax in flat_axes:
            ax.set_xlim(new_left, new_right)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
