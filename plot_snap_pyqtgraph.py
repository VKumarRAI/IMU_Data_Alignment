"""
Interactive impulse alignment viewer using PyQtGraph.
All three sensors aligned to dominant snap (global gyro-magnitude peak).
Controls: scroll to zoom, right-click+drag to pan, drag crosshair for ms readout.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

GYRO_SCALE = 10.0
ACC_SCALE = 800.0


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def load_adi(path):
    df = pd.read_csv(path)
    gx = df["X_GYRO_LWR"].values.astype(np.int64) / (GYRO_SCALE * 65536.0)
    gy = df["Y_GYRO_LWR"].values.astype(np.int64) / (GYRO_SCALE * 65536.0)
    gz = df["Z_GYRO_LWR"].values.astype(np.int64) / (GYRO_SCALE * 65536.0)
    ax = df["X_ACCL_LWR"].values.astype(np.int64) / (ACC_SCALE * 65536.0)
    ay = df["Y_ACCL_LWR"].values.astype(np.int64) / (ACC_SCALE * 65536.0)
    az = df["Z_ACCL_LWR"].values.astype(np.int64) / (ACC_SCALE * 65536.0)
    cntr = df["DATA_CNTR"].values.astype(np.int64)
    rollover = int(cntr.max()) + 10
    unwrap = np.zeros(len(cntr), dtype=np.int64)
    unwrap[0] = cntr[0]
    for i in range(1, len(cntr)):
        d = cntr[i] - cntr[i - 1]
        if d < -(rollover // 2):
            d += rollover
        unwrap[i] = unwrap[i - 1] + d
    t = (unwrap - unwrap[0]) / 10.0   # milliseconds
    return t, gx, gy, gz, ax, ay, az


def load_kernel(path):
    with open(path) as f:
        lines = f.readlines()
    hdr_idx = next(i for i, l in enumerate(lines) if "Rate_X" in l)
    headers = lines[hdr_idx].split()
    rows = []
    for l in lines[hdr_idx + 1:]:
        parts = l.split()
        if len(parts) == len(headers):
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
    data = np.array(rows)
    gx = data[:, headers.index("Rate_X")]
    gy = data[:, headers.index("Rate_Y")]
    gz = data[:, headers.index("Rate_Z")]
    ax = data[:, headers.index("Acc1_X")]
    ay = data[:, headers.index("Acc1_Y")]
    az = data[:, headers.index("Acc1_Z")]
    ns = data[:, headers.index("SecondFraction")]
    t = np.zeros(len(ns))
    for i in range(1, len(ns)):
        dt = ns[i] - ns[i - 1]
        if dt < 0:
            dt += 1e9
        t[i] = t[i - 1] + dt / 1e6   # milliseconds
    return t, gx, gy, gz, ax, ay, az


def load_micro(path):
    with open(path) as f:
        lines = f.readlines()
    ds_idx = next(i for i, l in enumerate(lines) if "DATA_START" in l)
    hdr = lines[ds_idx + 1].strip().split(",")
    df = pd.DataFrame(
        [l.strip().split(",") for l in lines[ds_idx + 2:] if l.strip()],
        columns=hdr,
    )
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "gyrox" in lc: col_map[c] = "gx"
        elif "gyroy" in lc: col_map[c] = "gy"
        elif "gyroz" in lc: col_map[c] = "gz"
        elif "accelx" in lc: col_map[c] = "ax"
        elif "accely" in lc: col_map[c] = "ay"
        elif "accelz" in lc: col_map[c] = "az"
    df = df.rename(columns=col_map)
    for c in ["gx", "gy", "gz", "ax", "ay", "az"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    gx = np.degrees(df["gx"].values.astype(float))
    gy = np.degrees(df["gy"].values.astype(float))
    gz = np.degrees(df["gz"].values.astype(float))
    ax = df["ax"].values.astype(float)
    ay = df["ay"].values.astype(float)
    az = df["az"].values.astype(float)
    tcol = next(c for c in df.columns if "time" in c.lower())
    ts = df[tcol].str.replace(r"(\.\d{6})\d+", r"\1", regex=True)
    dt = pd.to_datetime(ts, format="%m/%d/%y %H:%M:%S.%f", errors="coerce")
    t = (dt - dt.iloc[0]).dt.total_seconds().values * 1000.0   # milliseconds
    return t, gx, gy, gz, ax, ay, az


def main():
    adi_path = (
        newest("aditest6*.csv")
        or newest("ADI_6_TEST_*.csv")
        or newest("ADI_5_TEST_*.csv")
        or newest("IMUIMPULSE4_*.csv")
    )
    k_path = (
        newest("kerneltest6*.txt")
        or newest("KERNELTEST_6*.txt")
        or newest("KERNELTEST5*.txt")
        or newest("K26*.txt")
    )
    micro_path = (
        newest("microtest6*.csv")
        or newest("MICRO_TEST_6*.csv")
        or newest("MICRO_TEST_5*.csv")
        or newest("MICROIMPULSE*.csv")
        or newest("IMUMICRO.csv")
    )

    missing = [n for n, p in [("ADI", adi_path), ("Kernel", k_path), ("Micro", micro_path)] if not p]
    if missing:
        print("Missing files for:", ", ".join(missing))
        sys.exit(1)

    print(f"ADI:    {os.path.basename(adi_path)}")
    print(f"Kernel: {os.path.basename(k_path)}")
    print(f"Micro:  {os.path.basename(micro_path)}")

    adi_t, adi_gx, adi_gy, adi_gz, adi_ax, adi_ay, adi_az = load_adi(adi_path)
    k_t, k_gx, k_gy, k_gz, k_ax, k_ay, k_az = load_kernel(k_path)
    micro_t, micro_gx, micro_gy, micro_gz, micro_ax, micro_ay, micro_az = load_micro(micro_path)

    def _find_snap(gyro_axes, t_ms, noise_window=200, onset_sigma=8, min_post=500):
        """Find snap as the first significant gyro event (onset of motion).

        1. Estimate noise floor from the quietest 200ms window.
        2. Find where any gyro axis first exceeds onset_sigma * noise_std.
        3. Refine to the nearest local peak on the triggering axis.
        Uses only gyro (not accel) to avoid gravity-offset false triggers.
        """
        n = len(gyro_axes[0])
        # Samples in noise_window ms
        dt = np.median(np.diff(t_ms[:min(1000, n)]))
        if dt <= 0:
            dt = 1.0
        win = max(int(noise_window / dt), 50)

        # Find quietest window: sliding RMS across all gyro axes
        combo = np.zeros(n)
        for g in gyro_axes:
            combo += np.nan_to_num(g, nan=0.0) ** 2
        combo = np.sqrt(combo)

        best_rms = np.inf
        best_start = 0
        step = max(win // 4, 1)
        for s in range(0, max(n - win, 1), step):
            rms = float(np.mean(combo[s : s + win]))
            if rms < best_rms:
                best_rms = rms
                best_start = s

        # Noise std per axis from quietest window
        noise_stds = []
        for g in gyro_axes:
            seg = np.nan_to_num(g[best_start : best_start + win], nan=0.0)
            noise_stds.append(max(float(np.std(seg)), 0.01))

        # Find first sample where any axis exceeds threshold
        onset_idx = n - 1
        trigger_axis = 0
        for ai, (g, ns) in enumerate(zip(gyro_axes, noise_stds)):
            ga = np.abs(np.nan_to_num(g, nan=0.0))
            thr = onset_sigma * ns
            above = np.where(ga > thr)[0]
            if len(above) > 0 and above[0] < onset_idx:
                onset_idx = int(above[0])
                trigger_axis = ai

        # Refine: find the strongest peak across ALL gyro axes near onset
        search_end = min(onset_idx + int(50 / dt), n)  # look 50ms ahead
        best_peak_val = 0.0
        peak_idx = onset_idx
        for g in gyro_axes:
            ga = np.abs(np.nan_to_num(g, nan=0.0))
            seg = ga[onset_idx:search_end]
            if len(seg) == 0:
                continue
            local_peak = int(np.argmax(seg))
            if seg[local_peak] > best_peak_val:
                best_peak_val = seg[local_peak]
                peak_idx = onset_idx + local_peak

        # Ensure enough post-snap data
        if (n - peak_idx) < min_post:
            peak_idx = max(0, n - min_post)

        return peak_idx

    adi_snap = _find_snap([adi_gx, adi_gy, adi_gz], adi_t)
    k_snap = _find_snap([k_gx, k_gy, k_gz], k_t)
    micro_snap = _find_snap([micro_gx, micro_gy, micro_gz], micro_t)

    print(f"Snap indices: ADI={adi_snap} ({adi_t[adi_snap]:.1f}ms)  "
          f"Kernel={k_snap} ({k_t[k_snap]:.1f}ms)  "
          f"Micro={micro_snap} ({micro_t[micro_snap]:.1f}ms)")

    adi_t_al = adi_t - adi_t[adi_snap]
    k_t_al = k_t - k_t[k_snap]
    micro_t_al = micro_t - micro_t[micro_snap]

    # Post-snap data available per sensor (ms after snap)
    adi_end_ms   = float(adi_t_al[-1])
    k_end_ms     = float(k_t_al[-1])
    micro_end_ms = float(micro_t_al[-1])
    common_end   = min(adi_end_ms, k_end_ms, micro_end_ms)
    print(f"Post-snap data: ADI={adi_end_ms:.0f}ms  Kernel={k_end_ms:.0f}ms  Micro={micro_end_ms:.0f}ms")
    print(f"Common window:  -50 to {common_end:.0f} ms")

    sensors = [
        ("ADI577", adi_t_al, [adi_gx, adi_gy, adi_gz, adi_ax, adi_ay, adi_az], (30, 100, 220), adi_end_ms),
        ("Kernel 220", k_t_al, [k_gx, k_gy, k_gz, k_ax, k_ay, k_az], (220, 50, 50), k_end_ms),
        ("Microstrain 3DM", micro_t_al, [micro_gx, micro_gy, micro_gz, micro_ax, micro_ay, micro_az], (50, 180, 50), micro_end_ms),
    ]
    gyro_labels = ["Gyro X (deg/s)", "Gyro Y (deg/s)", "Gyro Z (deg/s)"]
    accel_labels = ["Accel X", "Accel Y", "Accel Z"]

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    def build_window(title, axis_indices, axis_labels):
        win = QtWidgets.QMainWindow()
        win.setWindowTitle(title)
        win.resize(1400, 560)

        central = QtWidgets.QWidget()
        win.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(4, 4, 4, 4)

        status = QtWidgets.QLabel("Hover over plot to read time in ms")
        status.setStyleSheet("font-size:13px; padding:2px 8px; background:#222; color:#eee;")
        vbox.addWidget(status)


        glw = pg.GraphicsLayoutWidget()
        vbox.addWidget(glw)

        # Store plot curves for opacity control: {sensor_name: [list of PlotDataItem]}
        curve_map = {s[0]: [] for s in sensors}

        plots = []
        for row_idx, ylabel in enumerate(axis_labels):
            axis_idx = axis_indices[row_idx]
            p = glw.addPlot(row=row_idx, col=0)
            p.setLabel("left", ylabel)
            if row_idx == len(axis_labels) - 1:
                p.setLabel("bottom", "Time relative to snap (ms)")
            p.showGrid(x=True, y=True, alpha=0.4)
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            if plots:
                p.setXLink(plots[0])

            p.addLegend(offset=(10, 5))
            for name, t_al, axes, color, end_ms in sensors:
                y = axes[axis_idx]
                pen = pg.mkPen(color=color, width=1.8)
                curve = p.plot(t_al, y, pen=pen, name=f"{name} (ends {end_ms:.0f}ms)")
                curve_map[name].append(curve)
                end_line = pg.InfiniteLine(
                    pos=end_ms,
                    angle=90,
                    pen=pg.mkPen(color=color, width=1, style=QtCore.Qt.DotLine),
                )
                p.addItem(end_line)

            snap_line = pg.InfiniteLine(
                pos=0,
                angle=90,
                pen=pg.mkPen("w", width=1.5, style=QtCore.Qt.DashLine),
                label="snap",
            )
            p.addItem(snap_line)
            plots.append(p)

        # Add opacity toggle button
        btn = QtWidgets.QPushButton("Cycle Opacity")
        vbox.addWidget(btn)
        # Opacity levels: 1.0 (opaque), 0.6 (medium), 0.3 (faint)
        opacities = [1.0, 0.6, 0.3]
        opacity_idx = {name: 0 for name in curve_map}

        def cycle_opacity():
            for name, curves in curve_map.items():
                # Cycle to next opacity for this sensor
                opacity_idx[name] = (opacity_idx[name] + 1) % len(opacities)
                alpha = opacities[opacity_idx[name]]
                for curve in curves:
                    c = curve.opts['pen'].color()
                    c.setAlphaF(alpha)
                    curve.setPen(pg.mkPen(color=c, width=1.8))

        btn.clicked.connect(cycle_opacity)

        plots[0].setXRange(-100, min(common_end, 1000.0), padding=0)

        v_lines = []
        for p in plots:
            vl = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen((220, 220, 100), width=1, style=QtCore.Qt.DotLine),
            )
            p.addItem(vl, ignoreBounds=True)
            v_lines.append(vl)

        def on_mouse_moved(pos, src):
            if src.sceneBoundingRect().contains(pos):
                mp = src.vb.mapSceneToView(pos)
                t_ms = mp.x()
                for vl in v_lines:
                    vl.setPos(t_ms)
                status.setText(f"  t = {t_ms:.3f} ms")

        for p in plots:
            p.scene().sigMouseMoved.connect(lambda pos, _p=p: on_mouse_moved(pos, _p))

        return win, plots

    gyro_win, gyro_plots = build_window(
        "Gyro Alignment  |  Scroll=zoom  Right-drag=pan  Hover=read ms",
        [0, 1, 2],
        gyro_labels,
    )
    accel_win, accel_plots = build_window(
        "Accel Alignment  |  Scroll=zoom  Right-drag=pan  Hover=read ms",
        [3, 4, 5],
        accel_labels,
    )

    # Sync zoom/pan between gyro and accel windows.
    accel_plots[0].setXLink(gyro_plots[0])

    gyro_win.show()
    accel_win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
