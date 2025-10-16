import time
import numpy as np
import os, json
from datetime import datetime

# ---- import your drivers ----
from Lib.scope import RigolDualScopes
from Lib.heater_bus import HeaterBus

# ---- use your existing globals if present, else set safe defaults ----
V_MIN  = globals().get("V_MIN", 0.10)
V_MAX  = globals().get("V_MAX", 4.90)
V_BIAS_INPUT = globals().get("V_BIAS_INPUT", 2.50)
V_BIAS_INTERNAL = globals().get("V_BIAS_INTERNAL", 2.50)
INPUT_HEATERS = globals().get("INPUT_HEATERS", [28,29,30,31,32,33,34])

SCOPE1_CHANNELS = globals().get("SCOPE1_CHANNELS", [1,2,3,4])
SCOPE2_CHANNELS = globals().get("SCOPE2_CHANNELS", [1,2,3])

SETTLE = globals().get("SETTLE", 0.010)     # use a tad slower, more stable read
READ_AVG = globals().get("READ_AVG", 3)     # average a few reads for SNR

# ---------- math utils ----------
def _poly_r2(x, y, deg):
    """Return R^2 of a degree-deg polynomial fit."""
    coeffs = np.polyfit(x, y, deg)
    yhat = np.polyval(coeffs, x)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    return 1.0 - ss_res/ss_tot, coeffs

def _curvature_index(v, i):
    """Finite-difference 2nd derivative magnitude (normalized), peak value."""
    dv = np.mean(np.diff(v))
    d2 = np.convolve(i, [1, -2, 1], mode='same') / (dv*dv + 1e-12)
    # ignore edges
    pad = max(3, int(0.02*len(v)))
    if len(d2) > 2*pad:
        d2 = d2[pad:-pad]
        v  = v[pad:-pad]
    idx = np.argmax(np.abs(d2))
    return float(np.abs(d2[idx])), float(v[idx])  # (curvature magnitude, V_at_peak)

# ---------- hardware helpers ----------
def _set_all_inputs(bus, value):
    bus.send({h: float(value) for h in INPUT_HEATERS})

def _clip(v):
    return float(np.clip(v, V_MIN, V_MAX))

# ---------- single-heater sweep ----------
def sweep_one_heater(bus, scope, heater_id, v_center=V_BIAS_INPUT, span=1.6, steps=151):
    """
    Sweep one input heater around v_center by 'span' volts, measure all PD channels.
    Returns:
        V: (steps,) swept voltages
        I: (steps, n_pd) PD readings
    """
    v_lo = _clip(v_center - span/2)
    v_hi = _clip(v_center + span/2)
    V = np.linspace(v_lo, v_hi, steps)

    # Set all inputs to bias first
    _set_all_inputs(bus, V_BIAS_INPUT)
    time.sleep(0.2)

    I = []
    for v in V:
        bus.send({heater_id: float(v)})
        time.sleep(SETTLE)
        pd = scope.read_many(avg=READ_AVG)  # shape: (n_pd,)
        I.append(pd)
    return V, np.asarray(I, float)

def analyze_trace(V, I_mat):
    """
    For a single heater sweep:
      - compute linear vs quadratic R^2 per PD,
      - curvature index per PD,
      - choose a candidate bias per PD (where curvature peaks),
      - pick the overall PD/channel that shows strongest nonlinearity.
    Returns a dict with per-PD stats and a suggested bias voltage.
    """
    n_pd = I_mat.shape[1]
    per_pd = []
    for k in range(n_pd):
        y = I_mat[:, k]
        r2_lin, _ = _poly_r2(V, y, 1)
        r2_qua, _ = _poly_r2(V, y, 2)
        nl_gain = float(max(0.0, r2_qua - r2_lin))  # how much quad beats linear
        curv_mag, v_star = _curvature_index(V, y)
        per_pd.append({
            "pd": k,
            "r2_linear": float(r2_lin),
            "r2_quadratic": float(r2_qua),
            "nonlinearity_gain": nl_gain,
            "curvature": float(curv_mag),
            "v_bias_candidate": float(v_star),
            "y_mean": float(np.mean(y)),
            "y_std": float(np.std(y)),
        })

    # pick PD with best (nonlinearity_gain * curvature) as a robust score
    scores = [p["nonlinearity_gain"] * p["curvature"] for p in per_pd]
    best_idx = int(np.argmax(scores))
    suggested_bias = per_pd[best_idx]["v_bias_candidate"]
    return {
        "per_pd": per_pd,
        "best_pd": int(per_pd[best_idx]["pd"]),
        "suggested_bias": float(suggested_bias),
        "score": float(scores[best_idx]),
    }

def find_nonlinear_biases(mesh_bias=None, span=1.6, steps=151, save_json=True, out_dir="nl_scan"):
    """
    Main routine:
      1) optionally set internal mesh biases,
      2) sweep each input heater,
      3) compute suggested bias per heater,
      4) save a JSON report (and return the dictionary).
    """
    os.makedirs(out_dir, exist_ok=True)
    scope = RigolDualScopes(SCOPE1_CHANNELS, SCOPE2_CHANNELS, serial_scope1=None)
    bus = HeaterBus()

    try:
        # 1) internal mesh bias (if provided)
        if mesh_bias:
            bus.send(mesh_bias)
            time.sleep(0.3)

        # 2) keep other inputs at bias
        _set_all_inputs(bus, V_BIAS_INPUT)
        time.sleep(0.2)

        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "span": span,
            "steps": steps,
            "results": []
        }

        # 3) sweep each input heater
        for h in INPUT_HEATERS:
            print(f"[SWEEP] Heater {h} ...")
            V, I = sweep_one_heater(bus, scope, h, v_center=V_BIAS_INPUT, span=span, steps=steps)
            stats = analyze_trace(V, I)
            entry = {
                "heater": int(h),
                "suggested_bias": stats["suggested_bias"],
                "best_pd": stats["best_pd"],
                "score": stats["score"],
                "per_pd": stats["per_pd"],
                "V_min": float(V.min()),
                "V_max": float(V.max()),
            }
            report["results"].append(entry)

        # 4) choose final suggested bias per heater
        suggested = {r["heater"]: _clip(r["suggested_bias"]) for r in report["results"]}
        report["suggested_input_bias"] = suggested

        # 5) persistence
        if save_json:
            path = os.path.join(out_dir, f"nonlinearity_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[SAVED] {path}")

        return report

    finally:
        try:
            scope.close()
        except:
            pass
        try:
            bus.close()
        except:
            pass

# ---------- optional: two-heater interaction quick test ----------
def two_heater_interaction(bus, scope, h_i, h_j, v_i0, v_j0, delta=0.25, points=5):
    """
    Small grid around (v_i0, v_j0). Returns (Vi_grid, Vj_grid, I tensor)
    and an interaction estimate per PD from bilinear fit.
    """
    Vi = np.linspace(_clip(v_i0 - delta), _clip(v_i0 + delta), points)
    Vj = np.linspace(_clip(v_j0 - delta), _clip(v_j0 + delta), points)
    I = np.zeros((points, points, len(scope.channels)), float)

    for a, vi in enumerate(Vi):
        for b, vj in enumerate(Vj):
            # all inputs at bias
            _set_all_inputs(bus, V_BIAS_INPUT)
            # set the two
            bus.send({h_i: float(vi), h_j: float(vj)})
            time.sleep(SETTLE)
            I[a, b, :] = scope.read_many(avg=READ_AVG)

    # bilinear coefficient d for each PD via least squares on grid
    # I ~ a + b*Vi + c*Vj + d*Vi*Vj
    VI, VJ = np.meshgrid(Vi, Vj, indexing="ij")
    X = np.column_stack([np.ones(VI.size), VI.ravel(), VJ.ravel(), (VI*VJ).ravel()])
    d_coeffs = []
    for k in range(I.shape[2]):
        y = I[:, :, k].ravel()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        d_coeffs.append(float(beta[3]))
    return Vi, Vj, I, np.array(d_coeffs)

# ---------- convenience: apply suggested biases ----------
def apply_suggested_biases(bus, suggested_dict):
    """Write suggested per-heater biases."""
    bus.send({int(h): float(v) for h, v in suggested_dict.items()})
    time.sleep(0.3)
    return True

if __name__ == "__main__":
    # 1) run the sweep & analysis
    report = find_nonlinear_biases(span=1.6, steps=151, save_json=True)

    # 2) print a compact summary and how to set biases
    print("\nSuggested input biases (mid-fringe-ish):")
    for r in report["results"]:
        print(f"  Heater {r['heater']}: {report['suggested_input_bias'][r['heater']]:.3f} V "
              f"(best PD {r['best_pd']}, score={r['score']:.3g})")

    # Example: to actually apply them in your main experiment:
    # bus = HeaterBus()
    # apply_suggested_biases(bus, report["suggested_input_bias"])
    # bus.close()