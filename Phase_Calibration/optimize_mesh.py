#!/usr/bin/env python3
# optimize_mesh_by_search.py
# Search the heater-voltage space to improve effective rank (and related metrics).

import os, json, time, math, random
import numpy as np

# === your modules
from Lib.scope import RigolDualScopes
from Lib.heater_bus import HeaterBus
from .randomize_mesh import (
    metrics, save_mesh_bundle, characterize_matrix_differential,
    CAL_DIR, OUT_DIR, MESH_HEATERS, INPUT_HEATERS, INPUT_BIAS,
)

# =========================
# EDITABLE PARAMETERS
# =========================
# Start from an existing voltages JSON (recommended) OR set to "" to start random
START_VOLTAGES_JSON = ""   # e.g. "./meshes/mesh_20251018_113658_trial1_voltages.json"

# Hardware & measurement configs
AVG_READS   = 2            # more averaging = less noise, slower
SETTLE_S    = 0.05         # thermal settle after changing mesh
DELTA_COL   = 0.15         # bump   used inside characterize_matrix_differential

# Search strategy
N_ITERS     = 150          # total proposals to try
K_HEATERS   = 5            # heaters per proposal (small random subset)
STEP_INIT   = 0.40         # initial step size (volts)
STEP_MIN    = 0.1         # don’t shrink below this
ANNEAL_TAU  = 45.0         # temperature decay constant (iterations)
RESEED_EVERY = 35          # try a small random reseed every N iterations

# Safety
V_GUARD     = 0.01         # keep this margin inside heater vmin/vmax (from calibration)
V_FALLBACK  = (0.10, 4.90) # used if a heater lacks cal json

# Objective weights
W_EFF_RANK  = 1.0
W_MU        = 0.8        # column norm coefficient of variation (computed here)
W_CONDZ     = 0.4            # mutual_coherence
W_COLCORR   = 0.6          # mean_abs_colcorr
W_COLNORMCV = 0.4          # penalty on cond_z above 1

# =========================


EPS = 1e-12

def _compute_cond_z(M):
    Mz = (M - M.mean()) / (M.std() + EPS)
    s = np.linalg.svd(Mz, compute_uv=False)
    return float(s.max() / max(s.min(), EPS))

def _compute_mutual_coherence(M):
    Mz = (M - M.mean()) / (M.std() + EPS)
    cols = Mz / (np.linalg.norm(Mz, axis=0, keepdims=True) + EPS)
    G = cols.T @ cols
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))

def _compute_mean_abs_colcorr(M):
    Mz = (M - M.mean()) / (M.std() + EPS)
    C = np.corrcoef(Mz, rowvar=False)
    np.fill_diagonal(C, 0.0)
    return float(np.mean(np.abs(C)))

def _ensure_metrics(M, m):
    """Return a copy of m with required keys filled if missing."""
    m = dict(m) if m is not None else {}
    if "cond_z" not in m:
        m["cond_z"] = _compute_cond_z(M)
    if "mutual_coherence" not in m:
        m["mutual_coherence"] = _compute_mutual_coherence(M)
    if "mean_abs_colcorr" not in m:
        m["mean_abs_colcorr"] = _compute_mean_abs_colcorr(M)
    if "eff_rank" not in m:
        # robust eff_rank from z-scored SVD if absent
        Mz = (M - M.mean()) / (M.std() + EPS)
        s = np.linalg.svd(Mz, compute_uv=False)
        s = np.maximum(s, EPS)
        p = s / s.sum()
        m["eff_rank"] = float(np.exp(-(p * np.log(p)).sum()))
    return m

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def _balance_M(M):
    r = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12  # PD gains
    c = np.linalg.norm(M, axis=0, keepdims=True) + 1e-12  # input gains
    return (M / r) / c

def _get_heater_limits(heater_id):
    """Use calibration json to bound voltages; fall back if missing."""
    cal_path = os.path.join(CAL_DIR, f"heater_{heater_id:02d}.json")
    if os.path.exists(cal_path):
        cal = _load_json(cal_path)
        vmin = float(cal.get("vmin", V_FALLBACK[0])) + V_GUARD
        vmax = float(cal.get("vmax", V_FALLBACK[1])) - V_GUARD
    else:
        vmin, vmax = V_FALLBACK
    if vmin > vmax:  # in case of bad cal
        vmin, vmax = V_FALLBACK
    return vmin, vmax

def _clip_voltages(volts: dict):
    """Clip each heater to its safe range."""
    out = {}
    for h, v in volts.items():
        vmin, vmax = _get_heater_limits(int(h))
        out[int(h)] = float(np.clip(float(v), vmin, vmax))
    return out

def _init_voltages():
    if START_VOLTAGES_JSON and os.path.exists(START_VOLTAGES_JSON):
        cfg = _load_json(START_VOLTAGES_JSON)
        return _clip_voltages({int(k): float(v) for k, v in cfg["voltages"].items()})
    # else random within cal limits
    rng = np.random.default_rng()
    vols = {}
    for h in MESH_HEATERS:
        vmin, vmax = _get_heater_limits(h)
        vols[h] = float(rng.uniform(vmin, vmax))
    return vols

def _measure_M(scope, bus, avg=AVG_READS, settle=SETTLE_S):
    """Measure M using your existing differential method."""
    time.sleep(settle)
    return characterize_matrix_differential(
        scope, bus, input_heaters=INPUT_HEATERS,
        baseline_bias=INPUT_BIAS, delta_v=DELTA_COL, avg=avg, settle=0.010
    )

def _col_norm_cv(M):
    norms = np.linalg.norm(M, axis=0)
    return float(np.std(norms) / (np.mean(norms) + 1e-12))

def objective_from_metrics(m: dict, M: np.ndarray):
    """Higher is better."""
    score = 0.0
    score += W_EFF_RANK * float(m["eff_rank"])
    score -= W_MU       * float(m["mutual_coherence"])
    score -= W_COLCORR  * float(m["mean_abs_colcorr"])
    score -= W_COLNORMCV* _col_norm_cv(M)
    score -= W_CONDZ    * max(0.0, float(m["cond_z"]) - 1.0)
    return float(score)

def propose_neighbor(volts: dict, step: float, k: int):
    """Choose k heaters at random and nudge them by ±step."""
    hsel = random.sample(list(volts.keys()), k=min(k, len(volts)))
    new = dict(volts)
    for h in hsel:
        new[h] = new[h] + (step if random.random() < 0.5 else -step)
    return _clip_voltages(new)

def temperature(iter_idx):
    # simple exponential cooling; influences acceptance of small regressions
    return math.exp(-iter_idx / max(ANNEAL_TAU, 1e-6))

def maybe_accept(delta, T):
    # accept if improved; else accept with small probability exp(delta/T) when delta<0
    if delta >= 0:
        return True
    return random.random() < math.exp(delta / max(T, 1e-6))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scope = RigolDualScopes([1,2,3,4], [1,2,3], serial_scope1='HDO1B244000779')
    bus   = HeaterBus()

    try:
        # --- init
        curr_volt = _init_voltages()
        bus.send(curr_volt)
        M = _measure_M(scope, bus)
        Mb = _balance_M(M)
        m = _ensure_metrics(Mb, metrics(Mb))
        #m = _ensure_metrics(M, metrics(M))

        best_volt = dict(curr_volt)
        best_M, best_m = M.copy(), dict(m)
        best_score = objective_from_metrics(best_m, best_M)

        print(f"[init] eff_rank={best_m['eff_rank']:.3f}  mu={best_m['mutual_coherence']:.2f}  "
              f"colcorr={best_m['mean_abs_colcorr']:.2f}  cond_z={best_m['cond_z']:.2f}  "
              f"score={best_score:.3f}")

        step = STEP_INIT

        for it in range(1, N_ITERS+1):
            # Occasionally reseed a few heaters to escape local minima
            candidate_volt = dict(curr_volt)
            if RESEED_EVERY and it % RESEED_EVERY == 0:
                candidate_volt = propose_neighbor(candidate_volt, step*2, k=max(2, K_HEATERS//2))
            candidate_volt = propose_neighbor(candidate_volt, step, K_HEATERS)
            # Skip if proposal didn't actually change anything
            if candidate_volt == curr_volt:
                step = min(0.25, step * 1.2)   # bump step a bit
                continue

            bus.send(candidate_volt)
            M1 = _measure_M(scope, bus)
            M1b = _balance_M(M1)
            m1 = _ensure_metrics(M1b, metrics(M1b))
            #m1 = _ensure_metrics(M1, metrics(M1))

            s1 = objective_from_metrics(m1, M1b)

            # decide acceptance
            T = temperature(it)
            s0 = objective_from_metrics(m, Mb)
            delta = s1 - s0
            acc = maybe_accept(delta, T)

            print(f"[{it:03d}] eff_rank={m1['eff_rank']:.3f}  mu={m1['mutual_coherence']:.2f}  "
                  f"colcorr={m1['mean_abs_colcorr']:.2f}  cond_z={m1['cond_z']:.2f}  "
                  f"Δscore={delta:+.3f}  {'ACCEPT' if acc else 'reject'}  step={step:.3f} T={T:.3f}")

            if acc:
                curr_volt, M, m = candidate_volt, M1, m1
                Mb = M1
                if s1 > best_score:
                    best_score = s1
                    best_volt  = dict(curr_volt)
                    best_M     = M1.copy()
                    best_m     = dict(m1)

            # If no accepted move for a while, randomize around best
            if it % 5 == 0 and abs(delta) < 0.1:
                print("[restart] no improvement, exploring new region")
                curr_volt = {h: best_volt[h] + np.random.uniform(-0.2, 0.2)
                            for h in best_volt}
                curr_volt = _clip_voltages(curr_volt)
                bus.send(curr_volt)
                M = _measure_M(scope, bus)
                Mb = _balance_M(M)
                m = _ensure_metrics(Mb, metrics(Mb))
                continue


            # simple step-size adaptation: shrink slowly if we haven't improved lately
            if it % 10 == 0 and best_score - s1 > 0.2:  # no progress margin
                step = max(STEP_MIN, step * 0.8)

        # --- save best bundle
        meta = {
            "tag": "optimized",
            "seed": None,
            "metrics": best_m,
            "objective": {
                "weights": {
                    "eff_rank": W_EFF_RANK,
                    "mutual_coherence": W_MU,
                    "mean_abs_colcorr": W_COLCORR,
                    "col_norm_cv": W_COLNORMCV,
                    "cond_z_penalty": W_CONDZ,
                },
                "score": best_score
            }
        }
        paths = save_mesh_bundle(best_volt, M=best_M, meta=meta)
        print(f"\n[Done] Best eff_rank={best_m['eff_rank']:.3f}, score={best_score:.3f}")
        print("[Saved]", paths)

        # restore the best voltages to leave the chip in a good state
        bus.send(best_volt)

    finally:
        try: scope.close()
        except Exception: pass
        try: bus.close()
        except Exception: pass

if __name__ == "__main__":
    main()
