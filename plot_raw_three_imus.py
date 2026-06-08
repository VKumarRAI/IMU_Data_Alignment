import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GYRO_SCALE = 10.0
ACCEL_SCALE = 800.0


def newest(pattern):
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)), key=os.path.getmtime)
    return files[-1] if files else None


def load_adi577():
    burst_path = newest("ADIS1657x_Burst_*.csv")
    impulse2_path = newest(os.path.join("May12", "ADI_IMPULSE2_*.csv"))
    reglog_path = impulse2_path or newest("ADIMU1000HzTest_*.csv") or newest("ADIMU1000Hz_*.csv") or newest("IMUDATATESTSYNC_*.csv") or newest("IMU_DATA_TEST_ADI_*.csv")

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
    k26_path = newest(os.path.join("May12", "KERNELIMPULSETEST2*.txt")) or newest("K26*.txt") or newest("KERNEL_*.txt")
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
    ms_path = newest(os.path.join("May12", "MICRO_IMPULSE_2*.csv")) or newest("IMUMICRO.csv") or newest("MicroStrain.csv") or newest("IMUMICRO*.csv") or newest("IMUDATA_*.csv") or newest("SensorConnectData.csv")
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

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
    fig.suptitle("Raw IMU Data (Gyro & Accel) - All Three Sensors", fontsize=12)

    colors = {
        "ADI577": "#1f77b4",
        "Kernel 220": "#d62728",
        "Microstrain 3DM": "#2ca02c",
    }

    channels = [
        ("Gyro X (deg/s)", (adi_t * 1000.0, adi_gx), (k_t * 1000.0, k_gx), (ms_t * 1000.0, ms_gx)),
        ("Gyro Y (deg/s)", (adi_t * 1000.0, adi_gy), (k_t * 1000.0, k_gy), (ms_t * 1000.0, ms_gy)),
        ("Gyro Z (deg/s)", (adi_t * 1000.0, adi_gz), (k_t * 1000.0, k_gz), (ms_t * 1000.0, ms_gz)),
        ("Accel X (g)",    (adi_t * 1000.0, adi_ax), (k_t * 1000.0, k_ax), (ms_t * 1000.0, ms_ax)),
        ("Accel Y (g)",    (adi_t * 1000.0, adi_ay), (k_t * 1000.0, k_ay), (ms_t * 1000.0, ms_ay)),
        ("Accel Z (g)",    (adi_t * 1000.0, adi_az), (k_t * 1000.0, k_az), (ms_t * 1000.0, ms_az)),
    ]

    for i, (title, adi_pair, k_pair, ms_pair) in enumerate(channels):
        r = i // 3
        c = i % 3
        ax = axes[r, c]

        ax.plot(adi_pair[0], adi_pair[1], color=colors["ADI577"], lw=0.9, label="ADI577")
        ax.plot(k_pair[0], k_pair[1], color=colors["Kernel 220"], lw=0.9, label="Kernel 220")
        ax.plot(ms_pair[0], ms_pair[1], color=colors["Microstrain 3DM"], lw=0.9, label="Microstrain 3DM")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.94, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
