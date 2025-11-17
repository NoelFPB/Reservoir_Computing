#!/usr/bin/env python3
# mesh_mixing_test.py
#
# Check how well the 4-layer (28-heater) MDC-like mesh mixes 7 inputs,
# using direct linear transfer matrices instead of curvature.

import os
import time
import json
import numpy as np
from datetime import datetime

from Lib.DualBoard import DualAD5380Controller
from Lib.scope import RigolDualScopes

# ========= Config =========

# 4 layers × 7 heaters = 28 internal mesh heaters
MESH_HEATERS   = list(range(28))                 # 0..27
# 7 input heaters that feed the PDs
INPUT_HEATERS  = [28, 29, 30, 31, 32, 33, 34]    # 7 "ports"

OUT_DIR        = "mesh_mixing_tests"
os.makedirs(OUT_DIR, exist_ok=True)

# Safe voltage window for all heaters
VMIN, VMAX     = 0.1, 4.90

# Settle times
SETTLE_MESH    = 0.10    # after changing mesh voltages
SETTLE_INPUT   = 0.03    # after changing input heaters

AVG_READS      = 1       # scope averaging
EPS            = 1e-12

# How many random meshes to test
N_RANDOM_MESHES = 50

# Input patterns for 7 input heaters
# We'll use 7 "one-hot-ish" patterns:
#   For column j: heater j = HIGH, others = LOW.
INPUT_LOW      = 2.4
INPUT_HIGH     = 4.0

def build_input_patterns():
    """
    Build a list of 7 patterns, each is a length-7 list of voltages
    on INPUT_HEATERS. Pattern j excites primarily port j.
    """
    patterns = []
    n = len(INPUT_HEATERS)
    for j in range(n):
        v = [INPUT_LOW] * n
        v[j] = INPUT_HIGH
        patterns.append(v)
    return patterns

INPUT_PATTERNS = build_input_patterns()


# ========= Helpers =========

def set_channels(bus, mapping):
    """mapping: dict {channel: voltage}"""
    if not mapping:
        return
    chs = list(mapping.keys())
    vs  = [float(mapping[c]) for c in chs]
    bus.set(chs, vs)

def random_mesh_like_phases(rng):
    """
    Emulate random phase shifts on all 28 mesh heaters.
    Map random theta in [0, 2π) linearly to [VMIN, VMAX].
    """
    theta = rng.uniform(0.0, 2.0 * np.pi, size=len(MESH_HEATERS))
    voltages = VMIN + (VMAX - VMIN) * (theta / (2.0 * np.pi))
    return {h: float(voltages[i]) for i, h in enumerate(MESH_HEATERS)}

def effective_rank_from_singulars(s):
    """Effective rank of singular values s."""
    s = np.maximum(s, EPS)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def analyze_mixing(H):
    """
    H: (n_outputs, n_inputs) matrix of linear responses.
    Returns:
        rank_norm   in [0, 1]
        mu          mutual coherence in [0, 1]
        eff_rank    effective rank (absolute)
    """
    # Center each column (remove DC)
    Hc = H - H.mean(axis=0, keepdims=True)

    # Column-wise normalization (z-score)
    Hc /= (Hc.std(axis=0, keepdims=True) + EPS)

    # Singular values
    s = np.linalg.svd(Hc, compute_uv=False)
    eff_rank = effective_rank_from_singulars(s)
    rank_norm = eff_rank / H.shape[1]  # normalize by #inputs (7)

    # Mutual coherence: max |corr(i,j)|, i≠j
    cols = Hc / (np.linalg.norm(Hc, axis=0, keepdims=True) + EPS)
    G = cols.T @ cols               # Gram matrix (7×7)
    np.fill_diagonal(G, 0.0)        # ignore self-correlation
    mu = float(np.max(np.abs(G)))   # max off-diagonal magnitude

    return rank_norm, mu, eff_rank


def measure_linear_matrix(scope, bus):
    """
    For the current mesh configuration (MESH_HEATERS fixed),
    measure a 7×7 mixing matrix H of PD responses to 7 input patterns.

    Returns:
        H: (n_pd, 7) matrix, where n_pd = number of PD channels read.
    """
    cols = []
    n_inputs = len(INPUT_HEATERS)

    for j, pattern in enumerate(INPUT_PATTERNS):
        # Set input heaters for pattern j
        input_map = {INPUT_HEATERS[k]: pattern[k] for k in range(n_inputs)}
        set_channels(bus, input_map)
        time.sleep(SETTLE_INPUT)

        # Read all PD outputs (vector)
        y = scope.read_many(avg=AVG_READS).astype(float)
        cols.append(y)

    H = np.stack(cols, axis=1)  # shape: (n_pd, 7)
    return H


# ========= Main =========

def main():
    rng   = np.random.default_rng()
    scope = RigolDualScopes(
        [1, 2, 3, 4], [1, 2, 3],
        serial_scope1='HDO1B244000779'
    )
    bus   = DualAD5380Controller()

    best_H = None
    best_mesh = None
    best_rank_norm = -np.inf
    best_mu = None

    all_rank_norms = []
    all_mus = []

    try:
        # ---- Warmup ----
        print("[warmup] Sending dummy mesh and discarding first measurement ...")
        dummy_mesh = random_mesh_like_phases(rng)
        set_channels(bus, dummy_mesh)
        time.sleep(SETTLE_MESH)

        # Do one dummy matrix read to warm scopes
        _ = scope.read_many(avg=AVG_READS)

        # ---- Main random sampling loop ----
        print(f"[info] Testing {N_RANDOM_MESHES} random meshes ...\n")

        for i in range(1, N_RANDOM_MESHES + 1):
            # Random mesh (all 28 heaters)
            mesh_volt = random_mesh_like_phases(rng)
            set_channels(bus, mesh_volt)
            time.sleep(SETTLE_MESH)

            # Measure linear mixing matrix
            H = measure_linear_matrix(scope, bus)

            # Mixing metrics
            rank_norm, mu, eff_rank = analyze_mixing(H)

            all_rank_norms.append(rank_norm)
            all_mus.append(mu)

            is_new_best = rank_norm > best_rank_norm

            if is_new_best:
                best_rank_norm = rank_norm
                best_mu = mu
                best_H = H.copy()
                best_mesh = dict(mesh_volt)

            print(
                f"[mesh {i:03d}] rank_norm={rank_norm:.3f} "
                f"(eff_rank={eff_rank:.2f})  mu={mu:.3f}  "
                f"{'⇧ new best' if is_new_best else ''}"
            )

        # ---- Summary ----
        all_rank_norms_np = np.array(all_rank_norms)
        all_mus_np = np.array(all_mus)

        print("\n========== SUMMARY ==========")
        print(f"Tested meshes          : {N_RANDOM_MESHES}")
        print(f"rank_norm  mean±std    : {all_rank_norms_np.mean():.3f} ± {all_rank_norms_np.std():.3f}")
        print(f"mu         mean±std    : {all_mus_np.mean():.3f} ± {all_mus_np.std():.3f}")
        print(f"\nBest mesh:")
        print(f"  rank_norm            : {best_rank_norm:.3f}")
        print(f"  mu                   : {best_mu:.3f}")

        # ---- Save best result ----
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"mixing_best_{ts}"

        np.save(os.path.join(OUT_DIR, base + "_H.npy"), best_H)
        with open(os.path.join(OUT_DIR, base + "_mesh_voltages.json"), "w") as f:
            json.dump(
                {
                    "mesh_voltages": best_mesh,
                    "rank_norm": float(best_rank_norm),
                    "mu": float(best_mu),
                },
                f,
                indent=2
            )

        print(f"\n[done] Saved best H and mesh voltages to:")
        print(f"  {os.path.join(OUT_DIR, base + '_H.npy')}")
        print(f"  {os.path.join(OUT_DIR, base + '_mesh_voltages.json')}")

        print("\nHeuristics:")
        print("  - Well-mixed target: rank_norm close to 1.0, mu clearly below ~0.7.")
        print("  - If you never see rank_norm > 0.8 or mu < 0.8, the device is not behaving like a good random unitary.")

    finally:
        try:
            scope.close()
        except Exception:
            pass
        try:
            bus.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
