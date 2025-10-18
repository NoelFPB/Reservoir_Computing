#!/usr/bin/env python3
# quick_mesh_matrix_test.py — interactive visualization mode
import numpy as np, os, json
import matplotlib.pyplot as plt

# === 👇 EDIT THIS LINE ONLY 👇 ===
M_PATH = "./meshes/mesh_20251018_171531_optimized_M.npy"
# =================================

EPS = 1e-12

def effective_rank(s):
    s = np.maximum(s, EPS)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def metrics(M):
    Mz = (M - M.mean()) / (M.std() + EPS)
    s = np.linalg.svd(Mz, compute_uv=False)
    eff_rank = effective_rank(s)
    Ccols = np.corrcoef(Mz, rowvar=False)
    Crows = np.corrcoef(Mz, rowvar=True)
    np.fill_diagonal(Ccols, 0)
    np.fill_diagonal(Crows, 0)
    mean_abs_colcorr = np.mean(np.abs(Ccols))
    mean_abs_rowcorr = np.mean(np.abs(Crows))
    cols = Mz / (np.linalg.norm(Mz, axis=0, keepdims=True) + EPS)
    G = cols.T @ cols
    np.fill_diagonal(G, 0)
    mu = np.max(np.abs(G))
    return {
        "eff_rank": eff_rank,
        "max_sv": float(s.max()),
        "min_sv": float(s.min()),
        "mean_abs_colcorr": float(mean_abs_colcorr),
        "mean_abs_rowcorr": float(mean_abs_rowcorr),
        "mutual_coherence": float(mu),
        "cond_z": float(s.max() / max(s.min(), EPS))
    }

def orthogonality_tests(M):
    G = M.T @ M
    Gn = G / (np.mean(np.diag(G)) + EPS)
    offdiag = Gn - np.diag(np.diag(Gn))
    max_off = np.max(np.abs(offdiag))
    mean_off = np.mean(np.abs(offdiag))
    col_norms = np.linalg.norm(M, axis=0)
    cv = np.std(col_norms) / (np.mean(col_norms) + EPS)
    return {"gram_max_offdiag": max_off, "gram_mean_offdiag": mean_off, "col_norm_cv": cv, "Gn": Gn}

def symmetry_error(M):
    A = M - M.T
    fro_rel = np.linalg.norm(A, "fro") / (np.linalg.norm(M, "fro") + EPS)
    max_rel = np.max(np.abs(A)) / (np.max(np.abs(M)) + EPS)
    return {"symm_fro_err": float(fro_rel), "symm_max_err": float(max_rel)}

def test_matrix(M):
    s_raw = np.linalg.svd(M, compute_uv=False)
    cond_raw = float(s_raw.max() / (s_raw.min() + EPS))
    rank = np.sum(s_raw > 1e-3 * s_raw.max())
    return cond_raw, int(rank), s_raw

def closest_unitary_polar(M):
    H = M.conj().T @ M
    w, V = np.linalg.eigh((H + H.conj().T) / 2.0)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(np.maximum(w, EPS))) @ V.conj().T
    return M @ inv_sqrt

def eigen_phase_stats(U):
    evals, evecs = np.linalg.eig(U)
    thetas = np.mod(np.angle(evals), 2*np.pi)
    thetas.sort()
    N = len(thetas)
    d = np.diff(np.r_[thetas, thetas[0] + 2*np.pi])
    s = (N / (2*np.pi)) * d
    intens = (np.abs(evecs)**2).ravel()
    return thetas, s, intens

def wigner_surmise_pdf(s):
    return (32.0 / (np.pi**2)) * (s**2) * np.exp(-4.0 * (s**2) / np.pi)

def porter_thomas_like_pdf(y, nu=4.0, y_mean=1.0):
    from math import gamma
    a = (nu/2.0)
    c = (a**a) / gamma(a)
    z = y / max(y_mean, EPS)
    return c * (z**(a - 1.0)) * np.exp(-a * z) / max(y_mean, EPS)

def main():
    if not os.path.exists(M_PATH):
        print(f"[ERROR] File not found: {M_PATH}")
        return

    M = np.load(M_PATH)
    print(f"[Loaded] {M_PATH}")
    print(f"Shape: {M.shape}")

    m = metrics(M)
    ortho = orthogonality_tests(M)
    symm = symmetry_error(M)
    cond_raw, rank, s_raw = test_matrix(M)

    print("\n--- MATRIX METRICS ---")
    print(json.dumps(m, indent=2))
    print("\n--- ORTHOGONALITY ---")
    print(json.dumps({k:v for k,v in ortho.items() if k!='Gn'}, indent=2))
    print("\n--- SYMMETRY ---")
    print(json.dumps(symm, indent=2))
    print(f"\nNumeric rank: {rank}, Condition (raw): {cond_raw:.3f}")

    # --- Plot everything interactively ---
    fig, axs = plt.subplots(2, 3, figsize=(14,8))
    axs = axs.ravel()

    axs[0].imshow(np.abs(M), aspect="auto")
    axs[0].set_title("|M|"); axs[0].set_xlabel("Inputs"); axs[0].set_ylabel("Outputs")

    axs[1].imshow(np.angle(M), aspect="auto")
    axs[1].set_title("∠M (radians)")

    axs[2].imshow(ortho["Gn"], aspect="auto")
    axs[2].set_title("Normalized Gram (MᵀM / ⟨diag⟩)")

    axs[3].stem(np.arange(1,len(s_raw)+1), s_raw)
    axs[3].set_title("Singular values (raw)")

    Mz = (M - M.mean()) / (M.std() + EPS)
    s_z = np.linalg.svd(Mz, compute_uv=False)
    axs[4].stem(np.arange(1,len(s_z)+1), s_z)
    axs[4].set_title("Singular values (z-score)")

    axs[5].axis("off")
    axs[5].text(0, 0.5,
        f"eff_rank = {m['eff_rank']:.2f}\n"
        f"cond_z = {m['cond_z']:.2f}\n"
        f"mutual_coherence = {m['mutual_coherence']:.2f}\n"
        f"mean_abs_colcorr = {m['mean_abs_colcorr']:.2f}\n"
        f"symm_fro_err = {symm['symm_fro_err']:.2f}",
        fontsize=10)

    plt.suptitle("Measured Matrix Diagnostics", fontsize=14)
    plt.tight_layout()

    # ---- Polar-decomposition + random-matrix-style plots ----
    U = closest_unitary_polar(M)
    thetas, spacings, intens = eigen_phase_stats(U)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.hist(thetas, bins=36, range=(0,2*np.pi), density=True)
    plt.title("Eigen-phase distribution")

    plt.subplot(1,3,2)
    plt.hist(spacings, bins=30, range=(0,max(3.0,spacings.max())), density=True)
    xs = np.linspace(0, max(3.0, spacings.max()), 400)
    plt.plot(xs, wigner_surmise_pdf(xs))
    plt.title("Level spacing (Wigner overlay)")

    plt.subplot(1,3,3)
    y = intens
    y_mean = np.mean(y)
    plt.hist(y, bins=40, range=(0,np.percentile(y,99)), density=True)
    xs = np.linspace(0, np.percentile(y,99), 400)
    plt.plot(xs, porter_thomas_like_pdf(xs, nu=4.0, y_mean=y_mean))
    plt.title("Eigenvector intensities (ν=4 overlay)")
    plt.tight_layout()

    # --- show all plots at once ---
    plt.show()

if __name__ == "__main__":
    main()
