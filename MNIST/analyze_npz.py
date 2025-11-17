import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ======================================
# CONFIG — set your file here
# ======================================
NPZ_FILE = "K4_R7_G02_real.npz"   # <-- change this to the file you want to analyze

BANDS = 7
PD = 7

# ======================================
# Helpers
# ======================================
def load_npz(path):
    d = np.load(path)
    return d["X"], d["y"]

def split_masks(X, bands=BANDS, pd=PD):
    """
    Split features into masks by inferring K from X.shape.
    Assumes D = bands * pd * K.
    """
    N, D = X.shape
    group = bands * pd

    if D % group != 0:
        raise ValueError(f"Feature dimension {D} is not divisible by {group}. Cannot infer K.")

    K = D // group
    print(f"Detected K = {K} masks")

    masks = []
    for m in range(K):
        cols = []
        for b in range(bands):
            start = b * (K * pd) + m * pd
            cols.extend(range(start, start + pd))
        mask_X = X[:, cols]
        masks.append(mask_X)
    return masks

def ridge_acc(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))
    clf.fit(Xtr_s, ytr)

    acc_tr = clf.score(Xtr_s, ytr)
    acc_te = clf.score(Xte_s, yte)

    return acc_tr, acc_te, clf.alpha_

def mask_correlation(mask_list):
    K = len(mask_list)
    C = np.zeros((K, K))
    flats = [m.ravel() for m in mask_list]
    for i in range(K):
        for j in range(K):
            C[i, j] = np.corrcoef(flats[i], flats[j])[0, 1]
    return C

def plot_pca(X, title):
    pca = PCA()
    pca.fit(X)
    ev = pca.explained_variance_ratio_

    plt.figure(figsize=(4,3))
    plt.plot(ev, "o-")
    plt.title(title)
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ======================================
# MAIN
# ======================================
if __name__ == "__main__":
    print("Loading:", NPZ_FILE)
    X, y = load_npz(NPZ_FILE)
    print("Data shape:", X.shape, y.shape)

    # Overall classifier performance
    acc_tr, acc_te, alpha = ridge_acc(X, y)
    print("\n=== Full Feature Classification ===")
    print(f"Train accuracy: {acc_tr:.4f}")
    print(f"Test accuracy : {acc_te:.4f}")
    print(f"Chosen alpha  : {alpha}")

    # Split masks
    masks = split_masks(X)

    print("\n=== Mask-wise Shapes ===")
    for i, m in enumerate(masks):
        print(f"Mask {i}: {m.shape}")

    # Mask correlation
    print("\n=== Mask Correlation Matrix ===")
    C = mask_correlation(masks)
    print(C)

    # PCA on full dataset
    print("\n=== PCA on full feature matrix ===")
    plot_pca(X, f"PCA Spectrum — {NPZ_FILE}")

    # PCA per mask
    for i, m in enumerate(masks):
        plot_pca(m, f"PCA Mask {i}")

    # Accuracy vs number of masks
    print("\n=== Accuracy vs K (first K masks used) ===")
    K = len(masks)
    for k in range(1, K+1):
        Xk = np.hstack(masks[:k])
        acc_tr_k, acc_te_k, _ = ridge_acc(Xk, y)
        print(f"K={k}: train={acc_tr_k:.4f}, test={acc_te_k:.4f}")
