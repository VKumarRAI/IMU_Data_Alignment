import os
import numpy as np
import csv
from datetime import datetime

def load_adi(path):
    # X_GYRO_LWR, Y_GYRO_LWR, Z_GYRO_LWR, DATA_CNTR
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([int(row["X_GYRO_LWR"]), int(row["Y_GYRO_LWR"]), int(row["Z_GYRO_LWR"]), int(row["DATA_CNTR"])])
    data = np.array(data)
    gx = data[:, 0] / (10.0 * 65536.0)
    gy = data[:, 1] / (10.0 * 65536.0)
    gz = data[:, 2] / (10.0 * 65536.0)
    cntr = data[:, 3]
    unwrap = np.zeros(len(cntr))
    rollover = int(np.max(cntr)) + 10
    unwrap[0] = cntr[0]
    for i in range(1, len(cntr)):
        d = cntr[i] - cntr[i-1]
        if d < -(rollover // 2): d += rollover
        unwrap[i] = unwrap[i-1] + d
    t = (unwrap - unwrap[0]) / 10000.0
    return t, gx, gy, gz

def load_kernel(path):
    # Rate_X, Rate_Y, Rate_Z, SecondFraction
    with open(path, 'r') as f:
        lines = f.readlines()
    hdr_idx = next(i for i, l in enumerate(lines) if "Rate_X" in l)
    headers = lines[hdr_idx].split()
    rows = []
    for l in lines[hdr_idx+1:]:
        p = l.split()
        if len(p) >= len(headers):
            try: rows.append([float(x) for x in p])
            except: pass
    df_data = np.array(rows)
    # Map headers to indices
    h_idx = {h: i for i, h in enumerate(headers)}
    gx, gy, gz = df_data[:, h_idx["Rate_X"]], df_data[:, h_idx["Rate_Y"]], df_data[:, h_idx["Rate_Z"]]
    ns = df_data[:, h_idx["SecondFraction"]].astype(np.int64)
    t = np.zeros(len(ns))
    for i in range(1, len(ns)):
        dt = ns[i] - ns[i-1]
        if dt < 0: dt += int(1e9)
        t[i] = t[i-1] + dt / 1e9
    return t, gx, gy, gz

def load_micro(path):
    # Time, gyrox, gyroy, gyroz
    with open(path, 'r') as f:
        lines = f.readlines()
    ds_idx = next(i for i, l in enumerate(lines) if "DATA_START" in l)
    headers = lines[ds_idx+1].strip().split(",")
    h_idx = {h.lower(): i for i, h in enumerate(headers)}
    g_cols = [k for k in h_idx if "gyro" in k]
    gx_c = next(c for c in g_cols if "x" in c)
    gy_c = next(c for c in g_cols if "y" in c)
    gz_c = next(c for c in g_cols if "z" in c)
    
    rows_t = []
    rows_g = []
    for l in lines[ds_idx+2:]:
        p = l.strip().split(",")
        if len(p) < len(headers): continue
        rows_g.append([float(p[h_idx[gx_c]]), float(p[h_idx[gy_c]]), float(p[h_idx[gz_c]])])
        ts = p[h_idx["time"]].split(".")
        # Handle trailing digits by truncating to 6 decimals
        subsec = (ts[1] + "000000")[:6]
        dt = datetime.strptime(ts[0] + "." + subsec, "%m/%d/%y %H:%M:%S.%f")
        rows_t.append(dt)
    
    gx, gy, gz = np.degrees(np.array(rows_g)).T
    t0 = rows_t[0]
    t = np.array([(rt - t0).total_seconds() for rt in rows_t])
    return t, gx, gy, gz

def process():
    p_adi = "May12/ADI_IMPULSE2_2026-05-12T14-25-08_0000.csv"
    p_ker = "May12/KERNELIMPULSETEST2.txt"
    p_mic = "May12/MICRO_IMPULSE_2.csv"
    
    t_a, gxa, gya, gza = load_adi(p_adi)
    t_k, gxk, gyk, gzk = load_kernel(p_ker)
    t_m, gxm, gym, gzm = load_micro(p_mic)
    
    ma, mk, mm = [np.sqrt(x**2 + y**2 + z**2) for x,y,z in [(gxa,gya,gza), (gxk,gyk,gzk), (gxm,gym,gzm)]]
    tp_a, tp_k, tp_m = t_a[np.argmax(ma)], t_k[np.argmax(mk)], t_m[np.argmax(mm)]
    
    print(f"Peak Times (s): ADI={tp_a:.3f}, K220={tp_k:.3f}, Micro={tp_m:.3f}")
    
    grid = np.linspace(0.05, 1.5, 1451)
    
    def prep(t, m, tp):
        v = np.interp(grid, t - tp, m)
        return (v - np.mean(v)) / np.std(v)
    
    va, vk, vm = prep(t_a, ma, tp_a), prep(t_k, mk, tp_k), prep(t_m, mm, tp_m)
    
    for v1, v2, n1, n2 in [(vk, va, "K220", "ADI"), (vm, va, "Micro", "ADI"), (vm, vk, "Micro", "K220")]:
        c = np.correlate(v1, v2, "full")
        lag = (np.argmax(c) - (len(v1)-1)) * 1.0
        r = np.corrcoef(v1, v2)[0,1]
        print(f"{n1} vs {n2}: Lag={lag:.2f}ms, r={r:.4f}")

process()
