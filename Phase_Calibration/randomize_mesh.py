# mesh_apply_and_measure.py
import os, json, time, numpy as np
from datetime import datetime
from Lib.heater_bus import HeaterBus
from Lib.scope import RigolDualScopes
import csv


CAL_DIR = "calibration"
OUT_DIR = "meshes"
os.makedirs(OUT_DIR, exist_ok=True)

MESH_HEATERS = list(range(28))                 # 0..27
INPUT_HEATERS = [28, 29, 30, 31, 32, 33, 34]   # 7 inputs → 7 PDs

# your established input biases (keep them stable)
INPUT_BIAS = {28:1.732, 29:1.764, 30:2.223, 31:2.372, 32:1.881, 33:2.436, 34:2.852}

def metrics(M):
    """Return effective rank and a few quick quality metrics."""
    Mz = (M - M.mean()) / (M.std() + 1e-12)

    # singular values
    s = np.linalg.svd(Mz, compute_uv=False)
    s = np.maximum(s, 1e-12)
    p = s / s.sum()

    # Roy & Vetterli effective rank: exp(Entropy(p))
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))

    # correlations
    Ccols = np.corrcoef(Mz, rowvar=False)  # between columns
    Crows = np.corrcoef(Mz, rowvar=True)   # between rows
    np.fill_diagonal(Ccols, 0.0)
    np.fill_diagonal(Crows, 0.0)
    mean_abs_colcorr = float(np.mean(np.abs(Ccols)))
    mean_abs_rowcorr = float(np.mean(np.abs(Crows)))

    # mutual coherence between columns
    cols = Mz / (np.linalg.norm(Mz, axis=0, keepdims=True) + 1e-12)
    G = cols.T @ cols
    np.fill_diagonal(G, 0.0)
    mu = float(np.max(np.abs(G)))

    return {
        "eff_rank": eff_rank,
        "max_sv": float(s.max()),
        "min_sv": float(s.min()),
        "mean_abs_colcorr": mean_abs_colcorr,
        "mean_abs_rowcorr": mean_abs_rowcorr,
        "mutual_coherence": mu,
    }

def append_summary_row(csv_path, tag, seed, paths, m):
    header = ["tag","seed","eff_rank","mean_abs_colcorr","mean_abs_rowcorr",
              "mutual_coherence","max_sv","min_sv","M_npy","voltages_json"]
    row = [tag, seed, m["eff_rank"], m["mean_abs_colcorr"], m["mean_abs_rowcorr"],
           m["mutual_coherence"], m["max_sv"], m["min_sv"],
           paths.get("M_npy",""), paths.get("voltages_json","")]
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file: w.writerow(header)
        w.writerow(row)

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _voltage_for_random_phase_any(cal, rng):
    # Prefer dense unwrapped LUT span if present
    pg = np.array(cal.get("phi_grid_unwrapped", []), float)
    vg = np.array(cal.get("v_of_phi_unwrapped", []), float)
    if len(pg) >= 4 and len(vg) == len(pg):
        phi = rng.uniform(pg[0], pg[-1])
        return float(np.interp(phi, pg, vg))

    # Fallback to one-cycle LUT
    pg1 = np.array(cal.get("phi_grid_onecycle", []), float)
    vg1 = np.array(cal.get("v_of_phi_onecycle", []), float)
    if len(pg1) >= 4 and len(vg1) == len(pg1):
        phi = rng.uniform(0.0, 2*np.pi)
        return float(np.interp(phi, pg1, vg1))

    # Last resort: uniform in [vmin, vmax]
    return float(np.random.uniform(cal.get("vmin", 0.1), cal.get("vmax", 4.9)))

def build_random_mesh_voltages(cal_dir=CAL_DIR, heaters=MESH_HEATERS, seed=None):
    rng = np.random.default_rng(seed)
    volts = {}
    for h in heaters:
        cal = _load_json(os.path.join(cal_dir, f"heater_{h:02d}.json"))
        volts[h] = _voltage_for_random_phase_any(cal, rng)
    return volts

def save_mesh_bundle(volts, M=None, meta=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = meta.get("tag","") if meta else ""
    base = f"mesh_{ts}{('_'+tag) if tag else ''}"
    # save voltages JSON + CSV
    jpath = os.path.join(OUT_DIR, base + "_voltages.json")
    cpath = os.path.join(OUT_DIR, base + "_voltages.csv")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"voltages": volts, "meta": meta or {}}, f, indent=2)
    with open(cpath, "w", encoding="utf-8") as f:
        f.write("heater,voltage\n")
        for k in sorted(volts):
            f.write(f"{k},{volts[k]:.6f}\n")

    paths = {"voltages_json": jpath, "voltages_csv": cpath}

    if M is not None:
        npy = os.path.join(OUT_DIR, base + "_M.npy")
        csv = os.path.join(OUT_DIR, base + "_M.csv")
        np.save(npy, M)
        np.savetxt(csv, M, delimiter=",")
        paths.update({"M_npy": npy, "M_csv": csv})

    return paths

def characterize_matrix_differential(scope, bus, input_heaters=INPUT_HEATERS, *,
                                     baseline_bias=INPUT_BIAS, delta_v=0.15, avg=3, settle=0.01):
    """
    Measures an effective mixing matrix M (n_outputs x n_inputs).
    We read PDs at baseline bias, then bump one input heater by +delta_v,
    read again, and take the difference: column j = y_bumped - y_base.
    """
    # set baseline on inputs
    bus.send({h: float(baseline_bias[h]) for h in input_heaters})
    time.sleep(settle)
    y0 = scope.read_many(avg=avg).astype(float)   # (n_outputs,)

    cols = []
    for j, hj in enumerate(input_heaters):
        cmd = {h: float(baseline_bias[h]) for h in input_heaters}
        cmd[hj] = float(baseline_bias[hj] + delta_v)
        bus.send(cmd); time.sleep(settle)
        y1 = scope.read_many(avg=avg).astype(float)
        cols.append(y1 - y0)
    M = np.stack(cols, axis=1)  # (n_outputs x n_inputs)
    # restore baseline
    bus.send({h: float(baseline_bias[h]) for h in input_heaters})
    return M

def apply_and_measure(tag=None, seed=None,
                      serial_scope1="HDO1B244000779",
                      scope1_ch=[1,2,3,4], scope2_ch=[1,2,3],
                      avg=3):
    # 1) build random mesh voltages from your calibrations
    volts = build_random_mesh_voltages(seed=seed)

    # 2) open hardware, apply mesh, and measure M
    scope = RigolDualScopes(scope1_ch, scope2_ch, serial_scope1=serial_scope1)
    bus = HeaterBus()
    try:
        # apply random mesh voltages on internal heaters 0..27
        bus.send(volts)
        time.sleep(0.2)

        # measure random mixing matrix on the PDs (7x7 typically)
        M = characterize_matrix_differential(scope, bus, avg=avg)

        # ---- NEW: compute metrics, incl. effective rank ----
        m = metrics(M)
        print(f"[Metrics] eff_rank={m['eff_rank']:.2f} | "
              f"colcorr={m['mean_abs_colcorr']:.2f} | "
              f"rowcorr={m['mean_abs_rowcorr']:.2f} | "
              f"mu={m['mutual_coherence']:.2f}")

        # 3) save both voltages and M (+ store metrics in meta)
        meta = {
            "tag": tag or "",
            "seed": seed,
            "scope1_channels": scope1_ch,
            "scope2_channels": scope2_ch,
            "metrics": m,
        }
        paths = save_mesh_bundle(volts, M=M, meta=meta)

        # 3.5) append one-line summary CSV for fast review
        append_summary_row(os.path.join(OUT_DIR, "_summary.csv"),
                           tag or "", seed, paths, m)

        # 4) return everything so you can use it immediately
        return volts, M, paths

    finally:
        try: scope.close()
        except: pass
        try: bus.close()
        except: pass

if __name__ == "__main__":
    volts, M, paths = apply_and_measure(tag="trial1", seed=None)
    print("[Mesh] Voltages applied to heaters 0..27:")
    for k in sorted(volts):
        print(f"  H{k:02d}: {volts[k]:.3f} V")
    print("[Mesh] M shape:", M.shape)
    print("[Saved]", paths)
