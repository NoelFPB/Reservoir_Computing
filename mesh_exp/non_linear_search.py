#!/usr/bin/env python3
# nonlinear_bayes_opt.py
# Bayesian optimization of mesh voltages to maximize
# rank (effective_rank) and minimize mutual coherence.

import os, json, time, numpy as np
from datetime import datetime
from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes

# ========= Config =========
MESH_HEATERS   = list(range(28))                 # internal heaters (0..27)
INPUT_HEATERS  = [28, 29, 30, 31, 32, 33, 34]    # seven inputs to PDs
OUT_DIR        = "meshes"; os.makedirs(OUT_DIR, exist_ok=True)

# Safe voltage window for mesh heaters
VMIN, VMAX     = 0.1, 4.90

# Input probing around center (for curvature)
INPUT_CENTER   = 3.2
DELTA_V        = 0.1                              # small central diff step
SETTLE_MESH    = 0.10
AVG_READS      = 1
EPS            = 1e-12

# ========= Bayesian optimization hyperparams =========
BO_INIT_POINTS    = 5    # number of purely random initial meshes
BO_ITERS          = 60   # number of BO iterations (expensive evals)

# GP hyperparameters (you can tune)
GP_LENGTH_SCALE   = 0.8 * (VMAX - VMIN)   # RBF length-scale in volts
GP_VARIANCE       = 1.0                   # prior variance
GP_NOISE          = 1e-6                  # observation noise for GP

# Acquisition parameters
EI_XI             = 0.01                  # exploration parameter in EI
N_CANDIDATES      = 256                   # random candidate points per BO iteration



# ========= Helpers =========
# --- custom erf approximation (no scipy, no numpy.erf) ---
# Abramowitz & Stegun approximation
def erf(x):
    # save the sign of x
    sign = np.sign(x)
    x = np.abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911

    t = 1.0 / (1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign * y

def _clip(v):
    return float(np.clip(v, VMIN, VMAX))

def random_mesh(rng):
    """Random mesh dictionary: {heater: voltage}."""
    return {h: float(rng.uniform(VMIN, VMAX)) for h in MESH_HEATERS}

def set_dict(bus, d):
    chs, vs = list(d.keys()), list(d.values())
    bus.set(chs, vs)

def dict_to_vec(d):
    """Convert {heater: V} to a vector in the fixed order of MESH_HEATERS."""
    return np.array([d[h] for h in MESH_HEATERS], dtype=float)

def vec_to_dict(v):
    """Convert a voltage vector back to {heater: V}, clipped to [VMIN, VMAX]."""
    return {h: _clip(float(v[i])) for i, h in enumerate(MESH_HEATERS)}

def measure_curvature(scope, bus):
    """
    For the current mesh heater configuration, probe each input heater ±ΔV
    around INPUT_CENTER and estimate second derivative (curvature) for
    each PD output. Returns H with shape (n_outputs, n_inputs).
    """
    dv = float(DELTA_V)
    base_chs = list(INPUT_HEATERS)
    #base_vs  = [INPUT_CENTER] * len(base_chs)
    BASE_VECTORS = [
        [2.4, 3.1, 2.8, 3.6, 4.1, 3.3, 2.9]
    ]
    base_vs = BASE_VECTORS[0]


    # Baseline at center
    bus.set(base_chs, base_vs)
    y0 = scope.read_many(avg=AVG_READS).astype(float)

    cols = []
    for idx, h in enumerate(INPUT_HEATERS):
        # +dv and -dv only on the current input heater
        vp = base_vs.copy()
        vm = base_vs.copy()
        vp[idx] = base_vs[idx] + dv
        vm[idx] = base_vs[idx] - dv

        # y_plus
        bus.set(base_chs, vp)
        y_plus = scope.read_many(avg=AVG_READS).astype(float)

        # y_minus
        bus.set(base_chs, vm)
        y_minus = scope.read_many(avg=AVG_READS).astype(float)

        # Restore baseline before moving to next heater
        bus.set(base_chs, base_vs)

        # Central second difference with fixed dv
        col = (y_plus - 2 * y0 + y_minus) / (dv ** 2)
        cols.append(col)

    return np.stack(cols, axis=1)  # shape: (n_outputs, n_inputs)

def effective_rank(s):
    """Effective rank of singular values s."""
    s = np.maximum(s, EPS)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def score_from_curvature(H):
    """
    H: shape (n_outputs, n_inputs)
    Objective: maximize effective rank, minimize mutual coherence.
    """
    # 1) Normalize H column-wise (z-score per column)
    Hc = H - H.mean(axis=0, keepdims=True)
    Hc /= (Hc.std(axis=0, keepdims=True) + EPS)

    # 2) Effective rank of the normalized matrix
    s = np.linalg.svd(Hc, compute_uv=False)
    eff_rank = effective_rank(s)
    rank_norm = eff_rank / H.shape[1]    # normalized to [0, 1]

    # 3) Mutual coherence (max column correlation magnitude)
    cols = Hc / (np.linalg.norm(Hc, axis=0, keepdims=True) + EPS)
    G = cols.T @ cols
    np.fill_diagonal(G, 0.0)
    mu = float(np.max(np.abs(G)))        # in [0, 1], smaller is better

    # 4) Final mixing score: high rank, low coherence
    ALPHA = 0.5     # penalty weight for coherence
    score = rank_norm - ALPHA * mu

    # Optional: measure average curvature amplitude just for logging
    nl = min(12, np.mean(np.abs(H))) / 12.0

    print(f"non_linear {nl:.2f}  rank {rank_norm:.2f}  mu {mu:.2f}  score {score:.2f}")
    return score

# ========= Gaussian Process / Bayesian Optimization =========

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    """
    Squared-exponential (RBF) kernel.
    X1: (n1, d), X2: (n2, d)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dists = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2)
    return variance * np.exp(-0.5 * dists / (length_scale ** 2))

def gp_posterior(X_train, y_train, X_test,
                 length_scale=GP_LENGTH_SCALE,
                 variance=GP_VARIANCE,
                 noise=GP_NOISE):
    """
    Compute GP posterior mean and std at X_test.
    """
    X_train = np.atleast_2d(X_train)
    X_test  = np.atleast_2d(X_test)
    n_train = X_train.shape[0]

    K = rbf_kernel(X_train, X_train, length_scale, variance)
    K += (noise + 1e-8) * np.eye(n_train)

    # Inversion is fine for n_train ~ O(100)
    K_inv = np.linalg.inv(K)

    K_s  = rbf_kernel(X_train, X_test, length_scale, variance)  # (n_train, n_test)
    mu   = (K_s.T @ K_inv @ y_train).reshape(-1)                # (n_test,)

    # Full covariance; we just need diag
    K_ss = rbf_kernel(X_test, X_test, length_scale, variance)
    cov  = K_ss - K_s.T @ K_inv @ K_s
    var  = np.clip(np.diag(cov), 1e-12, np.inf)
    std  = np.sqrt(var)
    return mu, std

def expected_improvement(X_test, X_train, y_train, best_y,
                         xi=EI_XI):
    """
    Expected Improvement acquisition for maximization.
    """
    mu, sigma = gp_posterior(X_train, y_train, X_test)
    sigma = np.maximum(sigma, 1e-12)

    # EI = (mu - best - xi) * Phi(z) + sigma * phi(z)
    from math import sqrt, pi
    z = (mu - best_y - xi) / sigma

    # Standard normal PDF and CDF via erf
    phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


    ei = (mu - best_y - xi) * Phi + sigma * phi
    ei[sigma < 1e-12] = 0.0
    return ei

def propose_next_voltage(rng, X_train, y_train, bounds, best_y):
    """
    Random-search EI maximization within voltage bounds.
    bounds: array of shape (d, 2) with [vmin, vmax] for each heater.
    """
    d = bounds.shape[0]
    # Sample random candidates in voltage space
    cand = rng.uniform(bounds[:, 0], bounds[:, 1], size=(N_CANDIDATES, d))

    # Compute EI at candidates
    ei = expected_improvement(cand, X_train, y_train, best_y)
    idx = int(np.argmax(ei))
    return cand[idx, :]

# ========= Main =========

def main():
    rng   = np.random.default_rng()
    scope = RigolDualScopes([1,2,3,4], [1,2,3], serial_scope1='HDO1B244000779')
    bus   = DualAD5380Controller()

    best_volt, best_H, best_score = None, None, -np.inf

    dim    = len(MESH_HEATERS)
    bounds = np.array([[VMIN, VMAX]] * dim, dtype=float)

    # Storage for GP
    X_list = []
    y_list = []

    try:
        # ---- Warmup: discard first measurement ----
        print("[warmup] Discarding first measurement ...")
        dummy_volt = random_mesh(rng)
        set_dict(bus, dummy_volt)
        time.sleep(SETTLE_MESH)
        _ = measure_curvature(scope, bus)   # ignore

        # ---- Initial random evaluations for GP ----
        print(f"[init] Collecting {BO_INIT_POINTS} random meshes ...")
        for r in range(BO_INIT_POINTS):
            volts = random_mesh(rng)
            set_dict(bus, volts)
            time.sleep(SETTLE_MESH)

            H = measure_curvature(scope, bus)
            s = score_from_curvature(H)

            vec = dict_to_vec(volts)
            X_list.append(vec)
            y_list.append(s)

            if s > best_score:
                best_volt, best_H, best_score = dict(volts), H.copy(), s

            print(f"[init {r+1}/{BO_INIT_POINTS}] score={s:.4f}  "
                  f"{'⇧ new best' if s == best_score else ''}")

        X_train = np.vstack(X_list)
        y_train = np.array(y_list, dtype=float)

        print(f"[start BO] best initial score={best_score:.4f}")

        # ---- Bayesian Optimization loop ----
        for k in range(1, BO_ITERS + 1):
            # Propose next voltage vector by maximizing EI
            x_next = propose_next_voltage(rng, X_train, y_train, bounds, best_score)
            volts_next = vec_to_dict(x_next)

            # Evaluate on hardware
            set_dict(bus, volts_next)
            time.sleep(SETTLE_MESH)
            H_next = measure_curvature(scope, bus)
            s_next = score_from_curvature(H_next)

            # Update GP dataset
            X_train = np.vstack([X_train, x_next[None, :]])
            y_train = np.concatenate([y_train, [s_next]])

            improved = s_next > best_score
            if improved:
                best_volt, best_H, best_score = dict(volts_next), H_next.copy(), s_next

            print(f"[BO {k:03d}] score={s_next:.4f}  "
                  f"{'⇧ new best' if improved else ''}")

        # ---- Save & leave chip in best state ----
        set_dict(bus, best_volt)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"bo_best_{ts}"
        np.save(os.path.join(OUT_DIR, base + "_H.npy"), best_H)
        with open(os.path.join(OUT_DIR, base + "_voltages.json"), "w") as f:
            json.dump({"voltages": best_volt, "score": best_score}, f, indent=2)

        print(f"\n[Done] Best score={best_score:.4f}")
        for k in sorted(best_volt):
            print(f"  H{k:02d}: {best_volt[k]:.3f} V")

    finally:
        try:
            scope.close()
        except:
            pass
        try:
            bus.close()
        except:
            pass

if __name__ == "__main__":
    main()
