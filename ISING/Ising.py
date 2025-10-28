import time
import numpy as np
from typing import Optional, Sequence
from  Lib.scope import  RigolDualScopes
from Lib.heater_bus import HeaterBus
import itertools, numpy as np
import json
import matplotlib.pyplot as plt

N = 7   # Spin count for the MVP

# Two voltage levels that represent spins -1 and +1 (start conservative)
# This ones are like experimental based on the calibration pictures, dont really know whats better
NEG_SPIN  = 1.50   # volts for σ = -1
POS_SPIN = 4.50   # volts for σ = +1

V_MIN = 1.5
V_MAX = 4.5

CHANNELS_SCOPE1 = [1,2,3,4]   # edit to match your wiring
CHANNELS_SCOPE2 = [1,2,3]   # edit to match your wiring

# Heaters that encode the N spins (index as your board expects)
SPIN_HEATER_CH = [28,29,30,31,32,33,34]

# Choose which indices in the concatenated readout correspond to the N local-field PDs
PD_IDX: Sequence[int] = [0,1,2,3,4,5,6]  # length N
MESH_HEATER_CH = list(range(28))

# Averages for scope reads during identification / iteration
SCOPE_AVG_ID = 1
SCOPE_AVG_RUN = 1

# Ising loop
MAX_ITERS = 30
ANNEAL_NOISE_STD = 0.05
STOP_NOFLIP_STEPS = 15

# External field h (length N); keep zeros for MAXCUT-style tests
H_VEC = np.zeros(N, dtype=float)

# If you have a precomputed J (NxN), set it here to skip identification
TARGET_J: Optional[np.ndarray] = None

# ====================== FUNCTIONS ======================
def plot_energy(history, label=None):
    iters = [k for k,_E in history]
    Es    = [E for _k,E in history]
    plt.figure()
    plt.plot(iters, Es, marker='.', linewidth=1)
    if label: plt.title(f'Energy vs iteration – {label}')
    else:     plt.title('Energy vs iteration')
    plt.xlabel('Iteration'); plt.ylabel('Ising energy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_spin_raster(sigma_trace, title='Spin raster'):
    # sigma_trace: (T+1, N) with values in {-1,+1}
    A = (sigma_trace + 1)/2.0   # map {-1,+1} -> {0,1} for nice contrast
    plt.figure()
    plt.imshow(A.T, aspect='auto', interpolation='nearest')
    plt.xlabel('Iteration'); plt.ylabel('Spin index')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_ticks([0,1]); cbar.set_ticklabels(['-1','+1'])
    plt.tight_layout()

def plot_flip_activity(sigma_trace, title='Flip activity per spin'):
    # sum of absolute changes |sigma_k - sigma_{k-1}| per spin
    diffs = np.abs(np.diff(sigma_trace, axis=0))  # (T, N), elements in {0,2}
    flips = diffs.sum(axis=0) / 2.0               # number of flips per spin
    plt.figure()
    plt.bar(np.arange(len(flips)), flips)
    plt.xlabel('Spin index'); plt.ylabel('Flip count')
    plt.title(title)
    plt.tight_layout()

def run_digital_sim(J, h=None, sigma0=None, noise_std=0.0, iters=100):
    # Simple digital reference that mirrors your hardware rule
    N = J.shape[0]
    if h is None: h = np.zeros(N)
    if sigma0 is None: sigma = np.sign(np.random.randn(N)).astype(int); sigma[sigma==0]=1
    else: sigma = np.array(sigma0, int)
    hist = []; trace=[sigma.copy()]
    best_E = +1e9
    for k in range(iters):
        f = J @ sigma
        sigma_new = np.sign(h + f + np.random.normal(0, noise_std, size=N)).astype(int)
        sigma_new[sigma_new==0]=1
        E = float(-0.5 * sigma_new @ (J @ sigma_new) - h @ sigma_new)
        hist.append((k, E))
        if E < best_E: best_E = E
        sigma = sigma_new
        trace.append(sigma.copy())
    return {'history': hist, 'sigma_trace': np.array(trace, int), 'E_best': best_E}

def save_mesh_biases(mesh_voltages, path="mesh_biases.json"):
    """Save current mesh heater voltages to a JSON file."""
    with open(path, "w") as f:
        json.dump(mesh_voltages, f, indent=2)
    print(f"[SAVE] Mesh biases saved to {path}")

def load_mesh_biases(path="mesh_biases.json"):
    """Load mesh heater voltages from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    # convert keys back to int if they were saved as strings
    return {int(k): float(v) for k, v in data.items()}

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, int); b = np.asarray(b, int)
    return int(np.sum(a != b))

def pass_fail_report(J_eff, run_result, energy_tol=1e-3):
    # exact ground state of the machine you *measured*
    E_g, s_g = brute_force_ground(J_eff)

    E_best = run_result["E_best"]
    s_best = run_result["sigma_best"].astype(int)

    # compare also against the global flip (same cut)
    s_g_flip = -s_g
    ham1 = hamming(s_best, s_g)
    ham2 = hamming(s_best, s_g_flip)
    ham = min(ham1, ham2)

    passed = (abs(E_best - E_g) <= energy_tol)

    report = {
        "PASS": passed,
        "E_ground_exact": float(E_g),
        "E_best_optical": float(E_best),
        "energy_gap": float(E_best - E_g),
        "sigma_ground": s_g.tolist(),
        "sigma_best_optical": s_best.tolist(),
        "hamming_to_ground_or_flip": ham,
    }
    return report

def maxcut_chain_J(N):
    J = np.zeros((N,N))
    for i in range(N-1):
        J[i, i+1] = J[i+1, i] = -1.0   # anti-ferro on chain edges
    return J

# expected ground state: alternating ±1 (or its global flip)
def alternating_sigma(N, start=1):
    s = np.array([start if (i % 2 == 0) else -start for i in range(N)], int)
    return s

# expected ground energy for the chain (N-1 edges, each satisfied):
# H = -(1/2) * sum_edges J_ij * sigma_i * sigma_j = -(1/2) * (N-1)
def expected_chain_energy(N):
    return -(N-1)/2.0

def ising_energy(sigma, J, h=None):
    sigma = np.asarray(sigma)
    h = np.zeros(len(sigma)) if h is None else np.asarray(h)
    return float(-0.5 * sigma @ (J @ sigma) - h @ sigma) # @ is matrix multiplication

def brute_force_ground(J, h=None):
    N = J.shape[0]
    bestE, bestS = +1e9, None
    for bits in itertools.product([-1,1], repeat=N):
        E = ising_energy(bits, J, h)
        if E < bestE:
            bestE, bestS = E, np.array(bits, int)
    return bestE, bestS

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
    def measure_fields_signed(self, sigma, avg):
        # measure with current spins
        self.write_spins(sigma)
        f_plus = self.measure_fields(avg)

        # measure with inverted spins
        sigma_inv = -sigma
        self.write_spins(sigma_inv)
        f_minus = self.measure_fields(avg)

        # restore spins
        self.write_spins(sigma)

        # signed local field ∝ J σ
        return 0.5 * (f_plus - f_minus)

    def _spins_to_voltages(self, sigma: np.ndarray) -> np.ndarray:
        return np.where(sigma >= 0, POS_SPIN, NEG_SPIN).astype(float)

    def write_spins(self, sigma: np.ndarray):
        """Encode spins on the SPIN_HEATER_CH as two voltage levels (serial .send)."""
        assert len(sigma) == self.n
        v = self._spins_to_voltages(sigma)
        self.heater.send((SPIN_HEATER_CH, v.tolist()))
  
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
        print("[ID] Identifying coupling matrix ...")
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
        #print(J)
        print("Coupling Matrix Identification Done.")
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
        sigma_trace = [sigma.copy()]

        for k in range(max_iters):
            self.write_spins(sigma)                               # (1) write spins
            f = self.measure_fields_signed(sigma, avg=SCOPE_AVG_RUN)            # (2) measure f ≈ Jσ
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
            sigma_trace.append(sigma.copy())
            
            print(f"[ITER {k:03d}] E={E:.6f}  noflip={noflip}")

            if noflip >= stop_noflip_steps:
                print("[STOP] No flips for consecutive steps — converged/stalled.")
                break

        return {
            "sigma_final": sigma,
            "sigma_best": best_sigma,
            "E_final": float(self.energy(sigma, J, h)),
            "E_best": float(best_energy),
            "history": history,             # list of (k, E_k)
            "sigma_trace": np.array(sigma_trace, int)  # shape: (T+1, N)
        }


def main():
    # -------------------- 0) Hardware boot --------------------
    scopes = RigolDualScopes(channels_scope1=CHANNELS_SCOPE1,
                             channels_scope2=CHANNELS_SCOPE2,
                             serial_scope1=None)
    heater = HeaterBus()
    ctrl = PhotonicIsingMF(N, heater, scopes, PD_IDX)

    # -------------------- 0.1) Mesh baseline ------------------
    # Use a safe, linear-ish mid-bias and narrower rail to avoid saturation.
    mesh_voltages = {h: 2.50 for h in MESH_HEATER_CH}
    heater.send(mesh_voltages)
    print("Inital J")
    print(mesh_voltages)
    # -------------------- helpers ------------------------------
    def identify_J(avg_repeats=SCOPE_AVG_ID):
        """Wrap identify + return (J, col_norms)."""
        J = ctrl.identify_coupling_matrix(avg_repeats=avg_repeats)
        col_norms = np.linalg.norm(J, axis=0)
        return J, col_norms

    def active_mask_from_norms(col_norms, thresh=0.20):
        """Columns below thresh are treated as inactive (dead)."""
        return (col_norms >= thresh)

    def masked_loss(J_eff, J_target, active_mask=None):
        """Frobenius loss with optional column masking (ignores dead inputs)."""
        if active_mask is None or active_mask.all():
            D = J_eff - J_target
            return float(np.linalg.norm(D, 'fro')**2)
        # zero-out inactive columns in both matrices before computing loss
        J1 = J_eff.copy(); J2 = J_target.copy()
        J1[:, ~active_mask] = 0.0; J2[:, ~active_mask] = 0.0
        D = J1 - J2
        return float(np.linalg.norm(D, 'fro')**2)
    

    # =======================
# SPSA-based mesh programming (drop-in)
# =======================
    def program_mesh_spsa(
        *,
        heater,                       # HeaterBus()
        identify_J_fn,                # callable: identify_J(avg_repeats:int) -> (J, col_norms)
        masked_loss_fn,               # callable: masked_loss(J_eff, J_target, active_mask)->float
        J_target: np.ndarray,
        mesh_heaters: list,           # e.g. MESH_HEATER_CH
        mesh_voltages: dict,          # {heater_id: voltage} initial state (will be updated)
        active_cols: np.ndarray,      # boolean mask from your earlier ID
        vmin: float = 0.10,           # safe bounds (adjust to your cal)
        vmax: float = 4.90,
        iters: int = 25,              # total SPSA iterations
        a0: float = 0.25,             # initial step size (volts)
        c0: float = 0.12,             # initial perturbation magnitude for SPSA (volts)
        verify_every: int = 3,        # do a full ID (slow) every N SPSA iterations
        avg_fast: int = 1,            # fast (proxy) ID averaging
        avg_verify: int = 3,          # full verify ID averaging
        rel_eps: float = 5e-3,        # relative improvement threshold for early stop (~0.5%)
        time_budget_sec: float = 5*60 # runtime cap
    ):
        """
        SPSA update on ALL mesh heaters with cheap proxy loss most of the time,
        and occasional full identify_J() to verify & checkpoint best state.
        Returns: best_volt, best_J_eff, final_active_cols, best_loss
        """
        t0 = time.perf_counter()
        H = len(mesh_heaters)
        heaters_arr = np.array(mesh_heaters, int)

        # Helpers
        def clip_voltages(vdict):
            out = {}
            for h, v in vdict.items():
                out[h] = float(np.clip(float(v), vmin, vmax))
            return out

        def send_voltages(vdict):
            heater.send({int(h): float(v) for h, v in vdict.items()})

        # ---- Initial verify (establish baseline) ----
        J_eff, norms = identify_J_fn(avg_repeats=avg_verify)
        active_cols = (np.linalg.norm(J_eff, axis=0) >= max(1e-9, np.percentile(np.linalg.norm(J_eff, axis=0), 5))) & active_cols
        best_loss = masked_loss_fn(J_eff, J_target, active_cols)
        best_volt = dict(mesh_voltages)
        best_J    = J_eff.copy()
        last_verified_loss = best_loss
        print(f"[SPSA] init loss={best_loss:.4f} (iters={iters}, a0={a0:.2f}, c0={c0:.2f})")

        a = float(a0)
        c = float(c0)
        ALPHA = 0.95  # step decay
        BETA  = 0.98  # perturb decay

        prev_verified_J = J_eff.copy()
        stable_hits = 0

        for it in range(1, iters+1):
            # --- time guard ---
            if (time.perf_counter() - t0) > time_budget_sec:
                print("[SPSA] time budget reached; stopping.")
                break

            # Rademacher perturbation (+1 / -1 per heater)
            Delta = np.random.choice([-1.0, 1.0], size=H)

            # Build v_plus / v_minus
            v_plus  = dict(mesh_voltages)
            v_minus = dict(mesh_voltages)
            for idx, h in enumerate(heaters_arr):
                v_plus[h]  = v_plus[h]  + c * Delta[idx]
                v_minus[h] = v_minus[h] - c * Delta[idx]
            v_plus  = clip_voltages(v_plus)
            v_minus = clip_voltages(v_minus)

            # --- Proxy evaluations (fast IDs) ---
            send_voltages(v_plus)
            Jp, _ = identify_J_fn(avg_repeats=avg_fast)
            Lp = masked_loss_fn(Jp, J_target, active_cols)

            send_voltages(v_minus)
            Jm, _ = identify_J_fn(avg_repeats=avg_fast)
            Lm = masked_loss_fn(Jm, J_target, active_cols)

            # Gradient estimate: g_i = (Lp - Lm)/(2c * Delta_i)
            # We aggregate heater-wise; if denominator small, skip safely.
            denom = (2.0 * max(c, 1e-6))
            ghat = {}
            diff = (Lp - Lm) / denom
            for idx, h in enumerate(heaters_arr):
                ghat[h] = float(diff * (1.0 / max(Delta[idx], 1e-6)))

            # Update voltages
            v_new = dict(mesh_voltages)
            for h in heaters_arr:
                v_new[h] = v_new[h] - a * ghat[h]
            v_new = clip_voltages(v_new)
            send_voltages(v_new)
            mesh_voltages = v_new  # commit

            # Decay schedules
            a *= ALPHA
            c *= BETA

            # --- Occasional verify with full identify_J (slower, accurate) ---
            if (it % verify_every) == 0 or it == iters:
                J_eff, norms = identify_J_fn(avg_repeats=avg_verify)
                # Optionally refresh active set, but do it conservatively to avoid flapping
                # e.g., mask columns smaller than 20% of median norm
                col_norms = np.linalg.norm(J_eff, axis=0)
                thresh = 0.20 * max(1e-9, np.median(col_norms))
                new_active = (col_norms >= thresh)
                if np.any(new_active):
                    active_cols = new_active

                L_true = masked_loss_fn(J_eff, J_target, active_cols)
                rel_improve = (last_verified_loss - L_true) / max(1e-9, last_verified_loss)
                print(f"[SPSA] it={it:03d}  L_true={L_true:.4f}  Δrel={rel_improve*100:.2f}%  (a={a:.3f}, c={c:.3f})")

                # Track best
                if (L_true + 1e-9) < (best_loss - max(1e-4, rel_eps * best_loss)):
                    best_loss = L_true
                    best_volt = dict(mesh_voltages)
                    best_J    = J_eff.copy()

                # Stability-based early stop: J not changing anymore
                rel_J = np.linalg.norm(J_eff - prev_verified_J, 'fro') / (np.linalg.norm(prev_verified_J, 'fro') + 1e-9)
                if rel_J < 5e-3:  # <0.5% change
                    stable_hits += 1
                else:
                    stable_hits = 0
                prev_verified_J = J_eff.copy()

                # Plateau early stop (loss not improving relatively)
                if rel_improve < rel_eps:
                    # shrink step to try fine convergence; if still flat next time, stop
                    a = max(0.5 * a, 0.02)
                last_verified_loss = L_true

                if stable_hits >= 2:
                    print("[SPSA] early stop: J stabilized.")
                    break

        # Restore best found and return
        send_voltages(best_volt)
        final_J, _ = identify_J_fn(avg_repeats=avg_verify)
        print(final_J)
        final_loss = masked_loss_fn(final_J, J_target, active_cols)
        print(f"[SPSA] done. best_loss={best_loss:.4f}  final_loss={final_loss:.4f}")
        return best_volt, final_J, active_cols, best_loss


    # -------------------- 1) Identify optics -------------------
    J_eff, norms = identify_J()
    active_cols = active_mask_from_norms(norms, thresh=0.20)
    if not np.any(active_cols):
        print("[ABORT] All columns appear inactive (dead). Check wiring/bias.")
        heater.close(); scopes.close(); return
    print("[ID] column norms:", np.round(norms, 4))
    print(f"[ID] active columns: {np.where(active_cols)[0].tolist()}")

    # -------------------- 2) Verify engine vs J_eff ------------
    # PASSES_NEEDED = 1
    # MAX_TRIALS    = 20
    # best_res = None
    # passes = 0
    # for s in range(MAX_TRIALS):
    #     np.random.seed(s)
    #     res = ctrl.run(J=J_eff, noise_std=0.1, max_iters=MAX_ITERS)
    #     rep = pass_fail_report(J_eff, res, energy_tol=1e-3)
    #     print("\n=== PASS/FAIL vs J_eff ===")
    #     for k, v in rep.items(): print(f"{k}: {v}")
    #     if rep.get("PASS", False):
    #         passes += 1
    #         if best_res is None or res["E_best"] < best_res["E_best"]:
    #             best_res = res
    #         if passes >= PASSES_NEEDED:
    #             break

    # if passes < PASSES_NEEDED:
    #     print("\n[ABORT] Engine did not PASS vs identified J_eff. Recalibrate and retry.")
    #     heater.close(); scopes.close(); return

    # -------------------- 3) Target problem --------------------
    J_target = maxcut_chain_J(N)
    print("Expected chain ground energy:", expected_chain_energy(N))
    print("Alternating ground (example):", alternating_sigma(N, start=1))

    # -------------------- 4) Program mesh toward target --------
    # -------------------- 4) Program mesh toward target (SPSA) --------
    # best_volt, J_eff, active_cols, best_loss = program_mesh_spsa(
    #     heater=heater,
    #     identify_J_fn=lambda avg_repeats: identify_J(avg_repeats=avg_repeats),
    #     masked_loss_fn=lambda J_eff_, J_tgt_, mask_: masked_loss(J_eff_, J_tgt_, mask_),
    #     J_target=J_target,
    #     mesh_heaters=MESH_HEATER_CH,
    #     mesh_voltages=mesh_voltages,   # starts from your mid-bias {h:2.50}
    #     active_cols=active_cols,
    #     vmin=V_MIN, vmax=V_MAX,
    #     iters=25,
    #     a0=0.25, c0=0.12,
    #     verify_every=3,
    #     avg_fast=1,     # proxy ID (cheap)
    #     avg_verify=3,   # verify ID (accurate)
    #     rel_eps=5e-3,
    #     time_budget_sec=25*60
    # )

    # print(f"[PROG] SPSA best_loss={best_loss:.4f}")
    # save_mesh_biases(best_volt)


    # Bounded greedy coordinate sweeps w/ early stop, patience, and runtime cap.
    DELTA                = 0.1
    MAX_SWEEPS           = 8        # hard cap on sweeps
    IMPROVE_EPS          = 0.1     # minimal improvement to count
    SWEEP_PATIENCE       = 1        # sweeps allowed with no improvement
    ID_EVERY_HEATERS     = 3        # throttle heavy identify calls
    RUNTIME_BUDGET_SEC   = 1*60    # e.g., 25 minutes runtime cap
    t_start              = time.perf_counter()

    # Current baseline loss
    best_loss = masked_loss(J_eff, J_target, active_cols)
    best_volt = mesh_voltages.copy()
    best_J    = J_eff.copy()
    print(f"[PROG] start loss={best_loss:.4f}")

    no_progress_sweeps = 0

    # In this for we basically program the problem to the J
    for sweep in range(1, MAX_SWEEPS + 1):
        improved = False
        # random order across mesh heaters
        for i, h in enumerate(np.random.permutation(MESH_HEATER_CH), start=1):
            # runtime guard
            if (time.perf_counter() - t_start) > RUNTIME_BUDGET_SEC:
                print("[PROG] runtime budget reached; stopping.")
                break

            v0 = mesh_voltages[h]
            trials = [
                v0,
                float(np.clip(v0 + DELTA, V_MIN, V_MAX)),
                float(np.clip(v0 - DELTA, V_MIN, V_MAX)),
            ]

            # Local best tracking
            best_local_v = v0
            best_local_L = None
            best_local_J = None

            for v_try in trials:
                heater.send({h: v_try})
                # Throttle the heavy ID to reduce total time; reuse best_J otherwise
                if (i % ID_EVERY_HEATERS) == 0 or best_local_L is None:
                    J_try, norms_try = identify_J(avg_repeats=SCOPE_AVG_ID)
                    active_try = active_mask_from_norms(norms_try, thresh=0.20)
                    # retain original active set if new ID is too pessimistic
                    # (avoids flapping due to noise)
                    if np.any(active_try):
                        active_cols = active_try
                else:
                    J_try = best_J  # fallback to last known good J

                L_try = masked_loss(J_try, J_target, active_cols)

                if (best_local_L is None) or (L_try < best_local_L - IMPROVE_EPS):
                    best_local_L, best_local_v, best_local_J = L_try, v_try, J_try

            # Commit local best
            if best_local_v != v0:
                mesh_voltages[h] = best_local_v
                heater.send({h: best_local_v})

            # Update global best
            if best_local_L is not None and (best_local_L < best_loss - IMPROVE_EPS):
                best_loss = best_local_L
                best_volt = mesh_voltages.copy()
                best_J    = best_local_J.copy()
                improved  = True
                print(f"[PROG] sweep {sweep} heater {h} -> loss {best_loss:.4f}")

        # Early stopping logic
        if improved:
            no_progress_sweeps = 0
        else:
            no_progress_sweeps += 1
            if no_progress_sweeps > SWEEP_PATIENCE:
                print("[PROG] early stop: no further improvement.")
                break

        # Runtime guard after each sweep
        if (time.perf_counter() - t_start) > RUNTIME_BUDGET_SEC:
            print("[PROG] runtime budget reached; stopping.")
            break

    # Restore best mesh and finalize identification
    heater.send(best_volt)
    J_eff, norms = identify_J(avg_repeats=SCOPE_AVG_ID)
    final_L = masked_loss(J_eff, J_target, active_cols)
    print(f"[PROG] final loss={final_L:.4f}")
    print("[ID] final column norms:", np.round(norms, 4))
    # --- Save mesh if the programming succeeded ---
    save_mesh_biases(best_volt)

    # -------------------- 5) Solve programmed problem ----------
    best = None
    for noise in np.linspace(0.1, 0.05, 3):
        res = ctrl.run(J=J_target, noise_std=noise, max_iters=30)
        if best is None or res["E_best"] < best["E_best"]:
            best = res

    # Compare to digital ground
    E_g_t, s_g_t = brute_force_ground(J_target)
    print("\n=== TARGET (chain) reference ===")
    print("E_ground :", E_g_t)
    print("sigma_g  :", s_g_t)

    print("\n=== OPTICAL RUN (programmed mesh) ===")
    print("E_best  :", best["E_best"])
    print("sigma_b :", best["sigma_best"])

    # -------------------- ---------------------------
    # Your optical result
    opt_hist  = best['history']
    opt_trace = best['sigma_trace']

    plot_energy(opt_hist, label='Optical')
    plot_spin_raster(opt_trace, title='Optical spin raster')
    plot_flip_activity(opt_trace, title='Optical flip activity')

    # Optional: overlay a digital simulation for comparison (same J_target)
    sim = run_digital_sim(J_eff, iters=len(opt_hist), noise_std=0.1)
    plot_energy(sim['history'], label='Digital')
    plt.figure()
    plt.plot([k for k,_E in opt_hist],[E for _k,E in opt_hist], label='Optical')
    plt.plot([k for k,_E in sim['history']],[E for _k,E in sim['history']], label='Digital')
    plt.xlabel('Iteration'); plt.ylabel('Ising energy'); plt.title('Energy vs iteration (Optical vs Digital)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    plt.show()

    # -------------------- 6) Cleanup ---------------------------
    heater.close()
    scopes.close()

if __name__ == "__main__":
    main()
