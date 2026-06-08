"""
clean_all_imu_gui.py

Unified Tkinter GUI for ADI, Microstrain, and KERNEL cleaners.
It processes selected raw files and writes:
1) Individual cleaned CSVs (optional)
2) One combined "big" CSV in a shared schema for cross-sensor analysis

Usage:
  python clean_all_imu_gui.py
"""

from __future__ import annotations

from datetime import datetime
import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

import clean_adi_analog_check as adi_cleaner
import clean_micro_check as micro_cleaner
import clean_kernel_check as kernel_cleaner


def find_adi_candidates(folder: str) -> list[str]:
    files: list[str] = []
    # Search recursively so files inside sensor subfolders are discovered
    for pat in ("IMUData_AnalogCheck*.csv", "ADI_*.csv", "*ADI*.csv", "*adi*.csv"):
        files.extend(glob.glob(os.path.join(folder, "**", pat), recursive=True))
    files = [f for f in set(files) if not f.endswith("_clean.csv")]
    return sorted(files, key=os.path.getmtime)


def find_micro_candidates(folder: str) -> list[str]:
    patterns = [
        "MICRO_IMPULSE_2*.csv",
        "IMUMICRO.csv",
        "MicroStrain.csv",
        "IMUMICRO*.csv",
        "MICRO_TEST_6*.csv",
        "MICROIMPULSE*.csv",
        "IMUDATA_*.csv",
        "SensorConnectData.csv",
        "IMUReferenceTimeStamp*.csv",
        "IMU*Reference*.csv",
        "MICRO*.csv",
    ]
    files: list[str] = []
    # include a lowercase/more general micro pattern to catch names like 'Calibration_Micro.csv'
    patterns = patterns + ["*micro*.csv", "*Micro*.csv"]
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, "**", pat), recursive=True))
    files = [f for f in set(files) if not f.endswith("_clean.csv")]
    return sorted(files, key=os.path.getmtime)


def find_kernel_candidates(folder: str) -> list[str]:
    # Kernel cleaner's own finder is non-recursive; search recursively here
    patterns = [
        "KERNEL*.txt",
        "*KERNEL*.txt",
        "OrientationTest*.txt",
        "ORIENTATION*.txt",
    ]
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, "**", p), recursive=True))
    files = [f for f in set(files) if not f.endswith("_clean.csv")]
    return sorted(files, key=os.path.getmtime)


def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "X_GYRO_LWR_raw": "GX_raw",
        "Y_GYRO_LWR_raw": "GY_raw",
        "Z_GYRO_LWR_raw": "GZ_raw",
        "X_ACCL_LWR_raw": "AX_raw",
        "Y_ACCL_LWR_raw": "AY_raw",
        "Z_ACCL_LWR_raw": "AZ_raw",
    }
    out = out.rename(columns=rename_map)
    for c in ["GX_raw", "GY_raw", "GZ_raw", "AX_raw", "AY_raw", "AZ_raw"]:
        if c not in out.columns:
            out[c] = np.nan
    return out


def sensorize(df: pd.DataFrame, sensor_name: str, source_file: str) -> pd.DataFrame:
    out = normalize_raw_columns(df)
    out.insert(0, "sample_index", np.arange(len(out), dtype=int))
    out.insert(0, "source_file", source_file)
    out.insert(0, "sensor", sensor_name)
    keep_cols = [
        "sensor",
        "source_file",
        "sample_index",
        "time_s",
        "time_ms",
        "gyro_x_deg_s",
        "gyro_y_deg_s",
        "gyro_z_deg_s",
        "accel_x_g",
        "accel_y_g",
        "accel_z_g",
        "GX_raw",
        "GY_raw",
        "GZ_raw",
        "AX_raw",
        "AY_raw",
        "AZ_raw",
    ]
    return out[keep_cols]


def make_output_path(output_dir: str, input_path: str, suffix: str = "_clean.csv") -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    # Append real-world timestamp to cleaned filenames automatically
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(output_dir, f"{base}_clean_{stamp}.csv")


def make_timestamped_csv_name(base_name: str) -> str:
    # Use experiment date (YYYY-MM-DD) for combined filename (date-only)
    stamp = datetime.now().strftime("%Y-%m-%d")
    base = (base_name or "IMU_ALL_COMBINED_clean.csv").strip()
    if base.lower().endswith(".csv"):
        stem = base[:-4]
    else:
        stem = base
    if not stem:
        stem = "IMU_ALL_COMBINED_clean"
    return f"{stem}_{stamp}.csv"


def process_adi(path: str, ts32: bool) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    return adi_cleaner.build_clean_df(
        raw,
        gyro_scale=adi_cleaner.GYRO_SCALE_DEFAULT,
        accel_scale=adi_cleaner.ACCEL_SCALE_DEFAULT,
        ts32=ts32,
    )


def process_micro(path: str) -> pd.DataFrame:
    return micro_cleaner.build_clean_df_from_micro(path, fallback_samplerate=False)


def process_kernel(path: str) -> pd.DataFrame:
    return kernel_cleaner.build_clean_df_from_kernel(path)


def launch_gui() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Prefer user-named project subfolders when present
    raw_subdir = "raw_data_imus"
    clean_subdir = "clean_data_imus"
    default_input = os.path.join(script_dir, raw_subdir) if os.path.isdir(os.path.join(script_dir, raw_subdir)) else script_dir
    # Default output folder should be the project's `clean_data_imus` directory
    default_output = os.path.join(script_dir, clean_subdir)

    root = tk.Tk()
    root.title("Unified IMU Cleaner - ADI + Micro + KERNEL")
    root.geometry("1024x640")
    root.minsize(980, 580)
    root.configure(bg="#f4f6fb")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("Card.TFrame", background="#ffffff")
    style.configure("Header.TLabel", background="#f4f6fb", foreground="#10243e", font=("Segoe UI", 18, "bold"))
    style.configure("Sub.TLabel", background="#f4f6fb", foreground="#334e68", font=("Segoe UI", 10))
    style.configure("CardTitle.TLabel", background="#ffffff", foreground="#1f3b57", font=("Segoe UI", 11, "bold"))
    style.configure("TButton", font=("Segoe UI", 10))

    outer = ttk.Frame(root, padding=14)
    outer.pack(fill="both", expand=True)

    ttk.Label(outer, text="Unified IMU Cleaning Workbench", style="Header.TLabel").pack(anchor="w")
    ttk.Label(
        outer,
        text="Select ADI, Microstrain, and KERNEL files, clean each, and export one combined CSV.",
        style="Sub.TLabel",
    ).pack(anchor="w", pady=(0, 12))

    top_card = ttk.Frame(outer, style="Card.TFrame", padding=12)
    top_card.pack(fill="x", pady=(0, 10))

    input_dir_var = tk.StringVar(value=default_input)
    output_dir_var = tk.StringVar(value=default_output)
    combined_name_var = tk.StringVar(value="IMU_ALL_COMBINED_clean.csv")
    save_individual_var = tk.BooleanVar(value=True)
    adi_ts32_var = tk.BooleanVar(value=False)

    # Experiment counter file (persisted in project folder)
    counter_file = os.path.join(script_dir, "experiment_counter.txt")

    def load_experiment_counter() -> int:
        try:
            with open(counter_file, "r") as f:
                n = int(f.read().strip())
                return max(1, n)
        except Exception:
            return 1

    def save_experiment_counter(n: int) -> None:
        try:
            with open(counter_file, "w") as f:
                f.write(str(int(n)))
        except Exception:
            pass

    ttk.Label(top_card, text="Input Folder", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Entry(top_card, textvariable=input_dir_var, width=84).grid(row=1, column=0, columnspan=3, sticky="we", padx=(0, 8), pady=(2, 8))

    def browse_input_folder() -> None:
        d = filedialog.askdirectory(initialdir=input_dir_var.get() or script_dir)
        if d:
            input_dir_var.set(d)
            refresh_candidates()

    ttk.Button(top_card, text="Browse", command=browse_input_folder).grid(row=1, column=3, sticky="e", pady=(2, 8))

    ttk.Label(top_card, text="Output Folder", style="CardTitle.TLabel").grid(row=2, column=0, sticky="w")
    ttk.Entry(top_card, textvariable=output_dir_var, width=84).grid(row=3, column=0, columnspan=3, sticky="we", padx=(0, 8), pady=(2, 8))

    def browse_output_folder() -> None:
        d = filedialog.askdirectory(initialdir=output_dir_var.get() or script_dir)
        if d:
            output_dir_var.set(d)

    ttk.Button(top_card, text="Browse", command=browse_output_folder).grid(row=3, column=3, sticky="e", pady=(2, 8))

    ttk.Label(top_card, text="Combined CSV Name", style="CardTitle.TLabel").grid(row=4, column=0, sticky="w")
    ttk.Entry(top_card, textvariable=combined_name_var, width=56).grid(row=5, column=0, sticky="w", pady=(2, 0))
    ttk.Checkbutton(top_card, text="Save individual cleaned CSVs", variable=save_individual_var).grid(row=5, column=1, sticky="w", padx=10)
    ttk.Checkbutton(top_card, text="ADI TS_32", variable=adi_ts32_var).grid(row=5, column=2, sticky="w")

    # Visible experiment label showing the next experiment number
    exp_label_var = tk.StringVar(value=f"Experiment: exp{load_experiment_counter():03d}")
    ttk.Label(top_card, textvariable=exp_label_var, style="Sub.TLabel").grid(row=4, column=1, sticky="w")

    sensors_card = ttk.Frame(outer, style="Card.TFrame", padding=12)
    sensors_card.pack(fill="x", pady=(0, 10))

    ttk.Label(sensors_card, text="File Selection", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

    adi_var = tk.StringVar()
    micro_var = tk.StringVar()
    kernel_var = tk.StringVar()

    adi_combo = ttk.Combobox(sensors_card, textvariable=adi_var, width=82, state="readonly")
    micro_combo = ttk.Combobox(sensors_card, textvariable=micro_var, width=82, state="readonly")
    kernel_combo = ttk.Combobox(sensors_card, textvariable=kernel_var, width=82, state="readonly")

    ttk.Label(sensors_card, text="ADI").grid(row=1, column=0, sticky="w", pady=(2, 2))
    adi_combo.grid(row=1, column=1, sticky="we", padx=(8, 0), pady=(2, 2))

    ttk.Label(sensors_card, text="Microstrain").grid(row=2, column=0, sticky="w", pady=(2, 2))
    micro_combo.grid(row=2, column=1, sticky="we", padx=(8, 0), pady=(2, 2))

    ttk.Label(sensors_card, text="KERNEL").grid(row=3, column=0, sticky="w", pady=(2, 2))
    kernel_combo.grid(row=3, column=1, sticky="we", padx=(8, 0), pady=(2, 2))

    status_var = tk.StringVar(value="Ready")

    def set_combo(combo: ttk.Combobox, variable: tk.StringVar, files: list[str]) -> None:
        names = [os.path.basename(f) for f in files]
        combo["values"] = names
        if names:
            combo.current(0)
        else:
            variable.set("")

    adi_files: list[str] = []
    micro_files: list[str] = []
    kernel_files: list[str] = []

    def refresh_candidates() -> None:
        nonlocal adi_files, micro_files, kernel_files
        folder = input_dir_var.get() or script_dir
        if not os.path.isdir(folder):
            status_var.set("Input folder does not exist")
            return
        # If the chosen folder contains a dedicated raw-data subfolder, prefer it.
        search_root = folder
        if not os.path.basename(folder).lower().startswith(raw_subdir) and os.path.isdir(os.path.join(folder, raw_subdir)):
            search_root = os.path.join(folder, raw_subdir)
        # Primary search (prefer raw_data_imus when present)
        adi_files = find_adi_candidates(search_root)
        micro_files = find_micro_candidates(search_root)
        kernel_files = find_kernel_candidates(search_root)

        # Fallback 1: if nothing found, search the selected folder itself (recursive)
        if not any((adi_files, micro_files, kernel_files)):
            adi_files = find_adi_candidates(folder)
            micro_files = find_micro_candidates(folder)
            kernel_files = find_kernel_candidates(folder)

        # Fallback 2: if still nothing and selected folder isn't project root, search project root
        if not any((adi_files, micro_files, kernel_files)) and folder != script_dir:
            adi_files = find_adi_candidates(script_dir)
            micro_files = find_micro_candidates(script_dir)
            kernel_files = find_kernel_candidates(script_dir)
        set_combo(adi_combo, adi_var, adi_files)
        set_combo(micro_combo, micro_var, micro_files)
        set_combo(kernel_combo, kernel_var, kernel_files)
        status_var.set(
            f"Loaded candidates (search: {os.path.basename(search_root)}) - ADI: {len(adi_files)}, Micro: {len(micro_files)}, KERNEL: {len(kernel_files)}"
        )

    ttk.Button(sensors_card, text="Refresh File Lists", command=refresh_candidates).grid(row=4, column=1, sticky="e", pady=(8, 0))

    # Keep a primary action button near file selection so it is always visible,
    # even when the lower log panel is large on smaller displays.
    ttk.Button(
        sensors_card,
        text="Clean + Combine",
        command=lambda: on_clean_and_combine(),
    ).grid(row=4, column=0, sticky="w", pady=(8, 0))

    action_card = ttk.Frame(outer, style="Card.TFrame", padding=12)
    action_card.pack(fill="both", expand=True)

    log_text = tk.Text(action_card, height=12, wrap="word", background="#0f172a", foreground="#dbeafe", font=("Consolas", 10))
    log_text.pack(fill="both", expand=True)

    def log(msg: str) -> None:
        log_text.insert("end", msg + "\n")
        log_text.see("end")
        root.update_idletasks()

    def pick_path(files: list[str], chosen_name: str) -> str | None:
        for f in files:
            if os.path.basename(f) == chosen_name:
                return f
        return None

    def on_clean_and_combine() -> None:
        out_dir = output_dir_var.get().strip() or default_output
        os.makedirs(out_dir, exist_ok=True)

        # Experiment counter and real-world timestamp for filenames
        exp_num = load_experiment_counter()
        now = datetime.now()
        time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        date_stamp = now.strftime("%Y-%m-%d")
        exp_tag = f"exp{exp_num:03d}"

        # Individual cleaned files are routed to sensor-specific subfolders.
        sensor_dir_map = {
            "ADI": "ADI",
            "MICRO": "MicroStrain",
            "KERNEL": "Kernel",
        }

        # Combined file is routed to an experiment folder for this run.
        experiment_dir = os.path.join(out_dir, f"experiment_{exp_tag}_{date_stamp}")
        os.makedirs(experiment_dir, exist_ok=True)

        selected = []
        adi_path = pick_path(adi_files, adi_var.get()) if adi_var.get() else None
        micro_path = pick_path(micro_files, micro_var.get()) if micro_var.get() else None
        kernel_path = pick_path(kernel_files, kernel_var.get()) if kernel_var.get() else None

        if adi_path:
            selected.append(("ADI", adi_path))
        if micro_path:
            selected.append(("MICRO", micro_path))
        if kernel_path:
            selected.append(("KERNEL", kernel_path))

        if not selected:
            messagebox.showwarning("No files selected", "Please select at least one sensor file")
            return

        status_var.set("Cleaning selected files...")
        log("=== Starting clean + combine ===")

        cleaned_frames: list[pd.DataFrame] = []

        for sensor, path in selected:
            try:
                log(f"Processing {sensor}: {os.path.basename(path)}")
                if sensor == "ADI":
                    cleaned = process_adi(path, adi_ts32_var.get())
                elif sensor == "MICRO":
                    cleaned = process_micro(path)
                else:
                    cleaned = process_kernel(path)

                if save_individual_var.get():
                    base = os.path.splitext(os.path.basename(path))[0]
                    ind_name = f"{base}_clean_{exp_tag}_{time_stamp}.csv"
                    sensor_subdir = os.path.join(out_dir, sensor_dir_map.get(sensor, sensor))
                    os.makedirs(sensor_subdir, exist_ok=True)
                    out_path = os.path.join(sensor_subdir, ind_name)
                    cleaned.to_csv(out_path, index=False, float_format="%.6f")
                    log(f"  Saved cleaned file: {out_path}")

                cleaned_frames.append(sensorize(cleaned, sensor, os.path.basename(path)))
            except Exception as exc:
                messagebox.showerror("Processing error", f"{sensor} failed:\n{exc}")
                status_var.set("Error")
                log(f"ERROR in {sensor}: {exc}")
                return

        combined = pd.concat(cleaned_frames, ignore_index=True)
        # Combined filename: use user-provided base, append experiment counter and real-world time.
        base = combined_name_var.get().strip() or "IMU_ALL_COMBINED_clean.csv"
        if base.lower().endswith(".csv"):
            stem = base[:-4]
        else:
            stem = base
        combined_fname = f"{stem}_{exp_tag}_{time_stamp}.csv"
        combined_out = os.path.join(experiment_dir, combined_fname)
        combined.to_csv(combined_out, index=False, float_format="%.6f")

        # Increment and persist experiment counter for next run
        save_experiment_counter(exp_num + 1)
        # Update visible experiment label to next value
        try:
            exp_label_var.set(f"Experiment: exp{exp_num+1:03d}")
        except Exception:
            pass

        log(f"Saved combined CSV: {combined_out}")
        log(f"Total rows: {len(combined):,}")
        log("Done.")
        status_var.set(f"Completed: {os.path.basename(combined_out)}")
        messagebox.showinfo("Completed", f"Saved combined CSV:\n{combined_out}\n\nRows: {len(combined):,}")

    buttons = ttk.Frame(outer)
    buttons.pack(fill="x", pady=(8, 0))

    ttk.Button(buttons, text="Clean + Combine", command=on_clean_and_combine).pack(side="left")
    ttk.Button(buttons, text="Quit", command=root.destroy).pack(side="right")

    status = ttk.Label(outer, textvariable=status_var, relief="sunken", anchor="w")
    status.pack(fill="x", pady=(8, 0), ipady=4)

    refresh_candidates()
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
