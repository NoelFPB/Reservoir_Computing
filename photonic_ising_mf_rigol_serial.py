#!/usr/bin/env python3
# photonic_ising_mf_rigol_serial.py
#
# Minimal measurement–feedback Ising loop for your multiport MDC/MZI mesh,
# wired to your exact HeaterBus (serial .send) and RigolDualScopes (.read_many).
#
# -----------------------------------------------------------------------------

import time
import numpy as np
from typing import Optional, Sequence

from  Lib.scope import  RigolDualScopes
from Lib.heater_bus import HeaterBus

# ====================== USER CONFIG ======================

# Spin count for the MVP
N = 8

# Heaters that encode the N spins (index as your board expects)
SPIN_HEATER_CH: Sequence[int] = list(range(N))

# Two voltage levels that represent spins -1 and +1 (start conservative)
V_LOW  = 0.50   # volts for σ = -1
V_HIGH = 1.50   # volts for σ = +1

# Heater settling time after each write (seconds)
SETTLE_S = 0.03

# --- Scopes / PD mapping ---
# The readout order from RigolDualScopes.read_many(avg) is:
#   [scope1 channels in CHANNELS_SCOPE1 order] + [scope2 channels in CHANNELS_SCOPE2 order]
CHANNELS_SCOPE1 = [1,2,3,4]   # edit to match your wiring
CHANNELS_SCOPE2 = [1,2,3,4]   # edit to match your wiring

# Choose which indices in the concatenated readout correspond to the N local-field PDs
PD_IDX: Sequence[int] = [0,1,2,3,4,5,6,7]  # length N

# Averages for scope reads during identification / iteration
SCOPE_AVG_ID = 3
SCOPE_AVG_RUN = 2

# Ising loop
MAX_ITERS = 250
ANNEAL_NOISE_STD = 0.05
STOP_NOFLIP_STEPS = 10

# External field h (length N); keep zeros for MAXCUT-style tests
H_VEC = np.zeros(N, dtype=float)

# If you have a precomputed J (NxN), set it here to skip identification
TARGET_J: Optional[np.ndarray] = None

# ====================== END CONFIG ======================


def sign_no_zero(x: np.ndarray) -> np.ndarray:
    s = np.sign(x)
    s[s == 0] = 1
    return s


class PhotonicIsingMF:
    def __init__(self,
                 n: int,
                 heater: HeaterBus,
                 scopes: RigolDualScopes,
                 pd_idx: Sequence[int]):
        self.n = n
        self.heater = heater
        self.scopes = scopes
        self.pd_idx = list(pd_idx)
        assert len(self.pd_idx) == n, "PD_IDX must have length N"
        self.J_eff: Optional[np.ndarray] = None

    # -------------------- Hardware I/O --------------------

    def _spins_to_voltages(self, sigma: np.ndarray) -> np.ndarray:
        return np.where(sigma >= 0, V_HIGH, V_LOW).astype(float)

    def write_spins(self, sigma: np.ndarray):
        """Encode spins on the SPIN_HEATER_CH as two voltage levels (serial .send)."""
        assert len(sigma) == self.n
        v = self._spins_to_voltages(sigma)
        self.heater.send((SPIN_HEATER_CH, v.tolist()))
        time.sleep(SETTLE_S)

    def _read_all_pd(self, avg: int) -> np.ndarray:
        """Read all configured scope channels and return concatenated vector."""
        vals = self.scopes.read_many(avg=max(1, int(avg)))  # shape (len(CH1)+len(CH2),)
        return np.asarray(vals, dtype=float)

    def measure_fields(self, avg: int) -> np.ndarray:
        """
        Return an N-vector of local-field proxies using PD_IDX to pick channels
        from the concatenated scope read. Assumes signed readout (homodyne/stat VMAX).
        If you only get intensities (>=0), implement two-shot sign inference here.
        """
        all_pd = self._read_all_pd(avg)
        f = all_pd[self.pd_idx]
        return f

    # ---------------- System ID for J_eff -----------------

    def identify_coupling_matrix(self, avg_repeats: int = SCOPE_AVG_ID) -> np.ndarray:
        """
        Build an effective J_eff satisfying f ≈ J_eff * sigma by probing +/- basis patterns.
        Column i ≈ 0.5 * (f(+e_i) - f(-e_i)).
        """
        print("[ID] Identifying effective coupling matrix ...")
        n = self.n
        J = np.zeros((n, n), dtype=float)

        sigma_saved = np.ones(n, dtype=int)  # baseline to restore

        for i in range(n):
            # +e_i (all -1 except spin i = +1)
            sigma_plus = -np.ones(n, dtype=int)
            sigma_plus[i] = +1
            self.write_spins(sigma_plus)
            f_plus = np.zeros(n, dtype=float)
            for _ in range(avg_repeats):
                f_plus += self.measure_fields(avg=avg_repeats)
            f_plus /= avg_repeats

            # -e_i (all -1; i stays -1)
            sigma_minus = -np.ones(n, dtype=int)
            self.write_spins(sigma_minus)
            f_minus = np.zeros(n, dtype=float)
            for _ in range(avg_repeats):
                f_minus += self.measure_fields(avg=avg_repeats)
            f_minus /= avg_repeats

            col = 0.5 * (f_plus - f_minus)
            J[:, i] = col
            print(f"[ID] Col {i}: ||col||={np.linalg.norm(col):.4f}")

        np.fill_diagonal(J, 0.0)  # optional: zero self-coupling
        self.write_spins(sigma_saved)
        self.J_eff = J
        print("[ID] Done.")
        return J

    # ------------------- Ising Loop ----------------------

    @staticmethod
    def energy(sigma: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
        # H = -1/2 sigma^T J sigma - h^T sigma
        return float(-0.5 * sigma @ (J @ sigma) - h @ sigma)

    def run(self,
            sigma0: Optional[np.ndarray] = None,
            J: Optional[np.ndarray] = None,
            h: Optional[np.ndarray] = None,
            max_iters: int = MAX_ITERS,
            noise_std: float = ANNEAL_NOISE_STD,
            stop_noflip_steps: int = STOP_NOFLIP_STEPS):
        n = self.n
        if J is None:
            if self.J_eff is None:
                raise RuntimeError("J is None and J_eff not identified.")
            J = self.J_eff
        if h is None:
            h = np.zeros(n, dtype=float)

        sigma = sign_no_zero(np.random.randn(n)) if sigma0 is None else sign_no_zero(np.array(sigma0, int))

        best_sigma = sigma.copy()
        best_energy = self.energy(sigma, J, h)
        noflip = 0
        history = []

        for k in range(max_iters):
            self.write_spins(sigma)                               # (1) write spins
            f = self.measure_fields(avg=SCOPE_AVG_RUN)            # (2) measure f ≈ Jσ
            sigma_new = sign_no_zero(h + f +                     # (3) update
                                     np.random.normal(0.0, noise_std, size=n))
            E = self.energy(sigma_new, J, h)                      # (4) energy (using J)
            history.append((k, E))

            if E < best_energy:
                best_energy, best_sigma = E, sigma_new.copy()

            if np.array_equal(sigma_new, sigma):
                noflip += 1
            else:
                noflip = 0

            sigma = sigma_new
            print(f"[ITER {k:03d}] E={E:.6f}  noflip={noflip}")

            if noflip >= stop_noflip_steps:
                print("[STOP] No flips for consecutive steps — converged/stalled.")
                break

        return {
            "sigma_final": sigma,
            "sigma_best": best_sigma,
            "E_final": float(self.energy(sigma, J, h)),
            "E_best": float(best_energy),
            "history": history,
        }


def main():
    # --- Hardware ---
    scopes = RigolDualScopes(channels_scope1=CHANNELS_SCOPE1,
                             channels_scope2=CHANNELS_SCOPE2,
                             serial_scope1=None)  # set a serial if you need a specific unit as scope1
    heater = HeaterBus()

    ctrl = PhotonicIsingMF(N, heater, scopes, PD_IDX)

    # --- Identify or load J ---
    if TARGET_J is not None:
        ctrl.J_eff = np.array(TARGET_J, dtype=float)
        print("[J] Using provided TARGET_J.")
    else:
        ctrl.identify_coupling_matrix()

    # --- Run loop ---
    result = ctrl.run()

    print("\n=== RESULTS ===")
    print("E_final :", result['E_final'])
    print("E_best  :", result['E_best'])
    print("sigma_f :", result['sigma_final'])
    print("sigma_b :", result['sigma_best'])

    # --- Cleanup ---
    try:
        heater.close()
    except Exception:
        pass
    try:
        scopes.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
