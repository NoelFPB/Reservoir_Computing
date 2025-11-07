    # calibrate_phase.py
import json, os, time, math, argparse
import numpy as np
import matplotlib.pyplot as plt

# ==========================
import os, json, time, csv

try:
    from scipy.optimize import curve_fit  # optional: for cosine fit
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Your drivers
from Lib.scope import RigolDualScopes
from Lib.heater_bus import HeaterBus

# ---------------------------
# Helpers
# ---------------------------

def safe_sleep(dt):
    if dt <= 0:
        return
    if dt < 0.002:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < dt:
            pass
    else:
        time.sleep(dt)

def moving_avg(x, k=5):
    if k <= 1:
        return x
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(xp, ker, mode="valid")

def cosine_model(V, A, B, phi0, a, b, c):
    # I(V) ~= A + B * cos( (a V^2 + b V + c) + phi0 )
    return A + B * np.cos((a * V * V + b * V + c) + phi0)

def monotone_unwrap_phase(phi_mod):
    """
    Take phase in [0, 2π) and unwrap to a monotone increasing curve.
    """
    phi = np.unwrap(phi_mod)  # jumps of 2π removed
    # Shift to start near 0..2π for convenience
    phi -= phi.min()
    return phi

def build_inverse_lut(v_grid, phi_unwrapped, n_phi=1024):
    """
    Create monotone φ(V) and inverse V(φ) LUTs via interpolation.
    """
    # Ensure strictly increasing φ; if not, sort by φ
    order = np.argsort(phi_unwrapped)
    phi_sorted = phi_unwrapped[order]
    v_sorted = v_grid[order]

    # Remove duplicates in φ (needed for interp)
    dphi = np.diff(phi_sorted)
    keep = np.r_[True, dphi > 1e-6]
    phi_sorted = phi_sorted[keep]
    v_sorted = v_sorted[keep]

    # Map φ to [0, 2π] by modulo
    # We'll also keep the unwrapped curve for multi-cycle coverage
    total_span = phi_sorted[-1] - phi_sorted[0]
    if total_span < 2*np.pi*0.8:
        # If we didn't cover a full 2π, warn but proceed
        print("[WARN] Phase sweep didn’t cover a full 2π. Inversion may be coarse.")
    # Build φ grid across covered span
    phi_min, phi_max = phi_sorted[0], phi_sorted[-1]
    phi_grid = np.linspace(phi_min, phi_max, n_phi)
    v_of_phi = np.interp(phi_grid, phi_sorted, v_sorted)

    # Also build a single-cycle (0..2π) LUT by folding
    # Fold unwrapped φ to [0, 2π)
    phi_fold = np.mod(phi_sorted, 2*np.pi)
    order2 = np.argsort(phi_fold)
    phi_fold = phi_fold[order2]
    v_fold = v_sorted[order2]

    # De-duplicate after fold
    dphi2 = np.diff(phi_fold)
    keep2 = np.r_[True, dphi2 > 1e-6]
    phi_fold = phi_fold[keep2]
    v_fold = v_fold[keep2]

    phi_grid_one = np.linspace(0.0, 2*np.pi, n_phi)
    v_of_phi_one = np.interp(phi_grid_one, phi_fold, v_fold)

    return {
        "phi_unwrapped": phi_sorted.tolist(),
        "v_for_phi_unwrapped": v_sorted.tolist(),
        "phi_grid_unwrapped": phi_grid.tolist(),
        "v_of_phi_unwrapped": v_of_phi.tolist(),
        "phi_grid_onecycle": phi_grid_one.tolist(),
        "v_of_phi_onecycle": v_of_phi_one.tolist(),
    }

def estimate_phase_from_cosine(V, I):
    """
    Fit cosine_model to I(V). Returns unwrapped phase φ(V) over the swept V.
    """
    # Smooth a little to reduce noise
    I_s = moving_avg(I, 5)

    # Initial guesses
    A0 = float(np.median(I_s))
    B0 = float((np.max(I_s) - np.min(I_s)) / 2.0)
    phi0_0 = 0.0
    # Quadratic phase vs V initial guess: mostly linear-ish start
    a0, b0, c0 = 0.0, 2.0, 0.0

    p0 = [A0, max(B0, 1e-3), phi0_0, a0, b0, c0]

    popt, _ = curve_fit(cosine_model, V, I_s, p0=p0, maxfev=20000)
    A, B, phi0, a, b, c = popt
    # Compute phase modulo 2π
    phi_mod = (a * V * V + b * V + c + phi0) % (2*np.pi)
    phi_unwrapped = monotone_unwrap_phase(phi_mod)

    return {
        "params": {"A": A, "B": B, "phi0": phi0, "a": a, "b": b, "c": c},
        "phi_unwrapped": phi_unwrapped
    }

def estimate_phase_lut_only(V, I):
    """
    Build a monotone φ(V) by detecting fringes and integrating.
    No SciPy required.
    """
    # Smooth
    I_s = moving_avg(I, 7)
    # Normalize to [0,1] for stability
    I_s = (I_s - I_s.min()) / max(1e-9, (I_s.max() - I_s.min()))

    # Find local extrema by simple neighbor comparison
    maxima = (np.r_[False, (I_s[1:-1] > I_s[0:-2]) & (I_s[1:-1] > I_s[2:]), False])
    minima = (np.r_[False, (I_s[1:-1] < I_s[0:-2]) & (I_s[1:-1] < I_s[2:]), False])

    idx_ext = np.where(maxima | minima)[0]
    if len(idx_ext) < 4:
        # Fallback: monotone mapping via normalized cumulative arc-cos surrogate
        # Map intensity to pseudo-phase by acos on [0,π]
        phi_mod = np.arccos(2*I_s - 1.0)  # [0, π], not exact but monotone per half-cycle
        phi_unwrapped = monotone_unwrap_phase(phi_mod)
        return {"phi_unwrapped": phi_unwrapped, "extrema_idx": idx_ext.tolist()}

    # Build phase by counting fringes between extrema:
    # Assign π/2 at minima→maxima, another π/2 maxima→minima, etc.
    phi = np.zeros_like(I_s)
    current_phi = 0.0
    last_idx = 0
    for k, idx in enumerate(idx_ext):
        # Linearly ramp phase between last_idx and idx
        seg_len = max(1, idx - last_idx)
        ramp = np.linspace(0, math.pi/2, seg_len, endpoint=False)
        phi[last_idx:idx] = current_phi + ramp
        current_phi += math.pi/2
        last_idx = idx
    # Fill tail
    if last_idx < len(phi):
        seg_len = len(phi) - last_idx
        phi[last_idx:] = current_phi + np.linspace(0, math.pi/2, seg_len)

    phi_unwrapped = phi - phi.min()
    return {"phi_unwrapped": phi_unwrapped, "extrema_idx": idx_ext.tolist()}

# ---------------------------
# Hardware sweep (ALL CHANNELS)
# ---------------------------

def sweep_heater(heater_id, vmin, vmax, n_points, settle_s, reads_avg, scope, bus):
    """
    Sweep voltage on one heater, read ALL PD channels each step.
    Returns:
        V_grid: (n_points,)
        I_all : (n_points, n_channels)
    """
    V_grid = np.linspace(vmin, vmax, int(n_points))
    I_rows = []

    for V in V_grid:
        bus.send({heater_id: float(V)})
        safe_sleep(settle_s)
        pd_vals = scope.read_many(avg=int(reads_avg))   # shape ~ (n_channels,)
        I_rows.append(pd_vals.astype(float))

    I_all = np.vstack(I_rows)                           # (n_points, n_channels)
    return V_grid, I_all


# ---------------------------
# Main calibration routine (ALL CHANNELS)
# ---------------------------

def _fringe_visibility(y):
    # Michelson visibility: (Imax - Imin) / (Imax + Imin)
    ymax, ymin = float(np.max(y)), float(np.min(y))
    return (ymax - ymin) / max(1e-9, (ymax + ymin))

def calibrate_one_heater(heater_id, vmin, vmax, n_points, settle_s, reads_avg,
                         channel,  # kept for backward-compat; not used now
                         scope, bus, outdir="calibration"):
    os.makedirs(outdir, exist_ok=True)

    print(f"[Cal] Sweeping heater {heater_id}: {vmin:.2f}→{vmax:.2f} V, points={n_points}")
    V, I_all = sweep_heater(heater_id, vmin, vmax, n_points, settle_s, reads_avg, scope, bus)
    n_points, n_channels = I_all.shape
    print(f"[Cal] Read {n_channels} PD channels.")

    # Per-channel phase extraction
    per_ch = []
    for ch in range(n_channels):
        I = I_all[:, ch]
        I_d = moving_avg(I, 3)

        fit_params = None
        method = "lut_only"
        try:
            if SCIPY_OK:
                fit = estimate_phase_from_cosine(V, I_d)
                phi_unwrapped = fit["phi_unwrapped"]
                fit_params = fit["params"]
                method = "cosine_fit"
            else:
                raise RuntimeError("SciPy not available")
        except Exception as e:
            # fallback
            lut = estimate_phase_lut_only(V, I_d)
            phi_unwrapped = lut["phi_unwrapped"]

        per_ch.append({
            "channel": ch + 1,                     # 1-index for human readability
            "phi_unwrapped": phi_unwrapped,
            "method": method,
            "fit_params": fit_params,
            "visibility": _fringe_visibility(I_d),
            "I_raw": I.tolist()
        })

    # Pick best channel by visibility (or by |B| if you prefer when fit succeeded)
    best_idx = int(np.argmax([d["visibility"] for d in per_ch]))
    best = per_ch[best_idx]
    print(f"[Cal] Best channel by visibility: ch {best['channel']} (vis={best['visibility']:.3f})")

    # Build inverse LUT from best channel
    inv = build_inverse_lut(V, best["phi_unwrapped"], n_phi=2048)

    # Save JSON with ALL channels
    cal = {
        "heater_id": heater_id,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "n_points": int(n_points),
        "settle_s": float(settle_s),
        "reads_avg": int(reads_avg),
        "V_grid": V.tolist(),
        "n_channels": int(n_channels),
        "per_channel": [
            {
                "channel": d["channel"],
                "method": d["method"],
                "fit_params": d["fit_params"],
                "visibility": float(d["visibility"]),
                "phi_unwrapped": d["phi_unwrapped"].tolist(),
                "I_raw": d["I_raw"]
            } for d in per_ch
        ],
        "best_channel": int(best["channel"]),
        "phi_grid_unwrapped": inv["phi_grid_unwrapped"],
        "v_of_phi_unwrapped": inv["v_of_phi_unwrapped"],
        "phi_grid_onecycle": inv["phi_grid_onecycle"],
        "v_of_phi_onecycle": inv["v_of_phi_onecycle"],
    }
    outpath = os.path.join(outdir, f"heater_{heater_id:02d}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"[Cal] Saved LUT → {outpath}")

    # ---------------------------
    # Plots: all channels
    # ---------------------------
    plt.figure(figsize=(15, 4.5))

    # (1) Intensities for all channels
    plt.subplot(1, 3, 1)
    for ch in range(n_channels):
        y = I_all[:, ch]
        plt.plot(V, y, '.', alpha=0.35, label=f"ch{ch+1}" if ch < 9 else None)
        plt.plot(V, moving_avg(y, 7), '-', alpha=0.9)
    plt.xlabel("Voltage (V)")
    plt.ylabel("PD intensity")
    plt.title(f"Heater {heater_id} sweep (all channels)")
    plt.legend(ncol=3, fontsize=8)

    # (2) Unwrapped phase per channel (normalized to start at 0)
    plt.subplot(1, 3, 2)
    for d in per_ch:
        phi = np.asarray(d["phi_unwrapped"])
        phi = phi - phi.min()
        plt.plot(V[:len(phi)], phi, lw=1.3, label=f"ch{d['channel']}")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Unwrapped phase (rad)")
    plt.title("φ(V) per channel")

    # (3) Inverse LUT (one cycle) from best channel
    plt.subplot(1, 3, 3)
    pg = np.array(inv["phi_grid_onecycle"])
    vg = np.array(inv["v_of_phi_onecycle"])
    plt.plot(pg, vg, '-')
    plt.xlabel("Phase φ (rad, 0..2π)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Inverse V(φ) — best ch {best['channel']}")

    plt.tight_layout()
    png = os.path.join(outdir, f"heater_{heater_id:02d}.png")
    plt.savefig(png, dpi=160)
    print(f"[Cal] Saved plot → {png}")
    plt.close()

    return outpath

# ---------------------------
# Convenience: use LUTs at runtime
# ---------------------------

def load_calibration(cal_file):
    with open(cal_file, "r", encoding="utf-8") as f:
        return json.load(f)

def voltage_for_phase(cal, phi):
    """
    Given a heater calibration dict and target phase phi in [0, 2π),
    return the corresponding voltage via the one-cycle inverse LUT.
    """
    phi = float(phi) % (2*np.pi)
    pg = np.array(cal["phi_grid_onecycle"])
    vg = np.array(cal["v_of_phi_onecycle"])
    return float(np.interp(phi, pg, vg))



# Your 7 input biases (keep them stable during all calibrations)
INPUT_HEATERS = [28, 29, 30, 31, 32, 33, 34]
INPUT_BIAS = {
    28: 1.732,
    29: 1.764,
    30: 2.223,
    31: 2.372,
    32: 1.881,
    33: 2.436,
    34: 2.852,
}

MESH_HEATERS = list(range(0, 28))  # 0..27
#MESH_HEATERS = {0,1,5,8,9,11,12,15,16,19,23,26}

def _apply_biases(bus, bias_dict):
    if bias_dict:
        bus.send({int(k): float(v) for k, v in bias_dict.items()})

def _phi_span_from_json(cal_json):
    # use best channel’s unwrapped phi to compute span
    best = int(cal_json.get("best_channel", 1))
    per = cal_json.get("per_channel", [])
    rec = next((d for d in per if int(d["channel"]) == best), None)
    if rec is None:
        # fallback: span from stored grid
        pg = np.array(cal_json.get("phi_grid_unwrapped", []), float)
        return float(pg[-1] - pg[0]) if len(pg) else float("nan")
    phi = np.array(rec["phi_unwrapped"], float)
    return float(phi.max() - phi.min()) if phi.size else float("nan")

def batch_calibrate(
    heaters=MESH_HEATERS,
    *,
    vmin=0.10,
    vmax=4.90,
    points=200,
    settle=1,
    reads=5,
    outdir="calibration",
    resume=True,
    scope=None,
    bus = None,
    sleep_between=1,         # tiny pause between heaters
    mid_bias_others=0        # hold non-swept mesh heaters here (optional)
):
    """
    Calibrate all mesh heaters (0..27) one-by-one, saving JSONs in `outdir`
    and a summary CSV `calibration/_summary.csv`.
    """
    os.makedirs(outdir, exist_ok=True)

    try:
        print("[Batch] Applying stable input biases...")
        _apply_biases(bus, INPUT_BIAS)

        if mid_bias_others is not None:
            print("[Batch] Setting mesh heaters to mid-bias before starting...")
            bus.send({h: float(mid_bias_others) for h in heaters})
            time.sleep(0.2)

        results = []
        for h in heaters:
            cal_path = os.path.join(outdir, f"heater_{h:02d}.json")
            if resume and os.path.exists(cal_path):
                print(f"[Batch] Skip heater {h:02d} (exists).")
                # collect quick stats for summary
                with open(cal_path, "r", encoding="utf-8") as f:
                    cal = json.load(f)
                results.append({
                    "heater": h,
                    "json": cal_path,
                    "best_ch": cal.get("best_channel", None),
                    "phi_span": _phi_span_from_json(cal),
                    "n_points": cal.get("n_points", None),
                    "reads": cal.get("reads_avg", None),
                })
                continue

            print(f"[Batch] Calibrating heater {h:02d}...")
            # hold all mesh heaters (except 'h') at mid-bias to reduce drift
            if mid_bias_others is not None:
                cmd = {hh: float(mid_bias_others) for hh in heaters if hh != h}
                bus.send(cmd)

            # run per-heater calibration (uses ALL PD channels; picks best)
            outpath = calibrate_one_heater(
                heater_id=h,
                vmin=vmin, vmax=vmax,
                n_points=points,
                settle_s=settle,
                reads_avg=reads,
                channel=1,              # ignored by our multi-channel version
                scope=scope, bus=bus,
                outdir=outdir
            )

            with open(outpath, "r", encoding="utf-8") as f:
                cal = json.load(f)
            results.append({
                "heater": h,
                "json": outpath,
                "best_ch": cal.get("best_channel", None),
                "phi_span": _phi_span_from_json(cal),
                "n_points": cal.get("n_points", None),
                "reads": cal.get("reads_avg", None),
            })

            time.sleep(sleep_between)

        # Write a compact CSV summary
        csv_path = os.path.join(outdir, "_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["heater", "best_ch", "phi_span", "n_points", "reads", "json"])
            w.writeheader()
            for r in sorted(results, key=lambda x: x["heater"]):
                w.writerow(r)
        print(f"[Batch] Summary written → {csv_path}")

        # Quick diagnostic: warn heaters with small span (<~ 2π)
        bad = [r for r in results if (isinstance(r["phi_span"], (int,float)) and r["phi_span"] < 2*np.pi*0.9)]
        if bad:
            ids = ", ".join(f"{r['heater']:02d}" for r in bad)
            print(f"[Batch][WARN] Low phase-span (<~2π) heaters: {ids} — consider re-running with more points or a different PD channel.")

        return results

    finally:
        try: scope.close()
        except Exception: pass
        try: bus.close()
        except Exception: pass


def main():
    # Defaults
    args = argparse.Namespace(
        heater=31,
        vmin=0.10,
        vmax=4.90,
        points=200,
        settle=0.020,
        reads=4,
        channel=4,             # <- choose the PD channel you want
        outdir="calibration",
        scope1=None,
        scope2=None,
    )

    scope = RigolDualScopes([1,2,3,4], [1,2,3], serial_scope1=args.scope1 or 'HDO1B244000779')
    bus = HeaterBus()

    try:
        batch_calibrate(
            heaters=list(range(28)),  # 0..27
            vmin=0.0, vmax=5.0,
            points=100,               # 121–201 recommended
            settle=0.20,
            reads=3,
            scope=scope,
            bus = bus,
            outdir="calibration",
            resume=True,
            mid_bias_others=0

        )
    finally:
        try: scope.close()
        except Exception: pass
        try: bus.close()
        except Exception: pass

if __name__ == "__main__":
    main()
