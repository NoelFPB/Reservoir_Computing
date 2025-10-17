    # calibrate_phase.py
import json, os, time, math, argparse
import numpy as np
import matplotlib.pyplot as plt

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
# Hardware sweep
# ---------------------------

def sweep_heater(heater_id, vmin, vmax, n_points, settle_s, reads_avg, scope, bus, channel=1):
    """
    Sweep voltage on one heater, measure intensity on one PD channel.
    Returns V_grid, I (float arrays).
    """
    V_grid = np.linspace(vmin, vmax, int(n_points))
    I = []

    # Prepare send dict initially (don’t disturb others)
    for V in V_grid:
        bus.send({heater_id: float(V)})
        safe_sleep(settle_s)
        pd_vals = scope.read_many(avg=int(reads_avg))  # shape ~ (n_channels,)
        # channel is 1-indexed in your config; convert to 0-index
        I.append(float(pd_vals[channel - 1]))
    return V_grid, np.array(I)

# ---------------------------
# Main calibration routine
# ---------------------------

def calibrate_one_heater(heater_id, vmin, vmax, n_points, settle_s, reads_avg, channel,
                         scope, bus, outdir="calibration"):
    os.makedirs(outdir, exist_ok=True)

    print(f"[Cal] Sweeping heater {heater_id}: {vmin:.2f}→{vmax:.2f} V, points={n_points}, ch={channel}")
    V, I = sweep_heater(heater_id, vmin, vmax, n_points, settle_s, reads_avg, scope, bus, channel)

    # Detrend gentle (optional)
    I_d = moving_avg(I, 3)

    # Estimate phase
    if SCIPY_OK:
        try:
            fit = estimate_phase_from_cosine(V, I_d)
            phi_unwrapped = fit["phi_unwrapped"]
            fit_params = fit["params"]
            method = "cosine_fit"
        except Exception as e:
            print(f"[WARN] Cosine fit failed: {e}. Falling back to LUT-only.")
            lut = estimate_phase_lut_only(V, I_d)
            phi_unwrapped = lut["phi_unwrapped"]
            fit_params = None
            method = "lut_only"
    else:
        lut = estimate_phase_lut_only(V, I_d)
        phi_unwrapped = lut["phi_unwrapped"]
        fit_params = None
        method = "lut_only"

    # Build inverse LUTs
    inv = build_inverse_lut(V, phi_unwrapped, n_phi=2048)

    # Save JSON
    cal = {
        "heater_id": heater_id,
        "method": method,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "n_points": int(n_points),
        "settle_s": float(settle_s),
        "reads_avg": int(reads_avg),
        "channel": int(channel),
        "V_grid": V.tolist(),
        "I_raw": I.tolist(),
        "phi_unwrapped": inv["phi_unwrapped"],
        "v_for_phi_unwrapped": inv["v_for_phi_unwrapped"],
        "phi_grid_unwrapped": inv["phi_grid_unwrapped"],
        "v_of_phi_unwrapped": inv["v_of_phi_unwrapped"],
        "phi_grid_onecycle": inv["phi_grid_onecycle"],
        "v_of_phi_onecycle": inv["v_of_phi_onecycle"],
        "fit_params": fit_params,
    }
    outpath = os.path.join(outdir, f"heater_{heater_id:02d}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"[Cal] Saved LUT → {outpath}")

    # Quick plots
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(V, I, '.', alpha=0.6, label='raw')
    plt.plot(V, moving_avg(I, 7), '-', alpha=0.9, label='smoothed')
    plt.xlabel("Voltage (V)"); plt.ylabel("PD intensity"); plt.title(f"Heater {heater_id} sweep"); plt.legend()

    plt.subplot(1,3,2)
    plt.plot(V, phi_unwrapped, '-', lw=1.5)
    plt.xlabel("Voltage (V)"); plt.ylabel("Unwrapped phase (rad)"); plt.title("φ(V) unwrapped")

    plt.subplot(1,3,3)
    pg = np.array(inv["phi_grid_onecycle"])
    vg = np.array(inv["v_of_phi_onecycle"])
    plt.plot(pg, vg, '-')
    plt.xlabel("Phase φ (rad, 0..2π)"); plt.ylabel("Voltage (V)")
    plt.title("Inverse V(φ) (one cycle)")

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

# ---------------------------
# CLI
# ---------------------------

def main():
    # Defaults
    args = argparse.Namespace(
        heater=28,
        vmin=0.20,
        vmax=4.80,
        points=121,
        settle=0.020,
        reads=3,
        channel=3,             # <- choose the PD channel you want
        outdir="calibration",
        scope1=None,
        scope2=None,
    )

    scope = RigolDualScopes([1,2,3,4], [1,2,3], serial_scope1=args.scope1 or 'HDO1B244000779')
    bus = HeaterBus()

    try:
        calibrate_one_heater(
            heater_id=args.heater,
            vmin=args.vmin,
            vmax=args.vmax,
            n_points=args.points,
            settle_s=args.settle,
            reads_avg=args.reads,
            channel=args.channel,
            scope=scope,
            bus=bus,
            outdir=args.outdir
        )
    finally:
        try: scope.close()
        except Exception: pass
        try: bus.close()
        except Exception: pass

if __name__ == "__main__":
    main()
