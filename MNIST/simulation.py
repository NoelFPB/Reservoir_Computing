import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Config
# -----------------------------
ROW_BANDS   = 7      # 28x28 -> 7 x ROW_BANDS
K_VIRTUAL   = 2       # 1 = no masks; >1 = 1 baseline + (K-1) ±1 masks
MASK_SEED   = 42
TEST_SIZE   = 0.2
SEED        = 42
N_PER_CLASS = 3000    

# -----------------------------
# Helpers
# -----------------------------
def downsample_to_7xM(img2d: np.ndarray, M: int) -> np.ndarray:
    """28x28 -> 7xM using max pooling (columns in blocks of 4, rows split into M bands)."""
    assert img2d.shape == (28, 28)
    col_reduced = img2d.reshape(28, 7, 4).mean(axis=2)   # (28,7)
    bands = np.array_split(col_reduced, M, axis=0)
    return np.stack([b.mean(axis=0) for b in bands], axis=0)  # (M,7)

def make_balanced_subset(X, y, n_per_class=150, seed=42):
    rng = np.random.default_rng(seed)
    idxs = []
    for d in range(10):
        di = np.where(y == d)[0]
        pick = rng.choice(di, size=min(n_per_class, len(di)), replace=False)
        idxs.append(pick)
    idx = np.concatenate(idxs)
    rng.shuffle(idx)
    return X[idx], y[idx]

def hadamard_like_masks(n_masks, width, seed=0, zero_sum=True, orth_thresh=0.2):
    """Generate ±1 masks with near zero-sum and low cross-correlation."""
    rng = np.random.default_rng(seed)
    M = []
    ones = np.ones(width)
    while len(M) < n_masks:
        v = rng.choice([-1.0, 1.0], size=width)
        if zero_sum and abs(v.sum()) > 1:  # near zero DC
            continue
        if zero_sum and abs((v @ ones) / width) > 0.25:
            continue
        if all(abs(np.dot(v, m) / width) < orth_thresh for m in M):
            M.append(v)
    return np.array(M, dtype=float)

def build_direct_with_masks(images_28x28, row_bands=7, k_virtual=1, mask_seed=42):
    """
    Downsample to (row_bands x 7). For each 7-wide chunk (each row),
    duplicate it k_virtual times with masks: [ones] + (k_virtual-1) ±1 masks.
    Output shape: [N, row_bands * 7 * k_virtual]
    """
    # Prepare masks for the 7-wide chunks
    if k_virtual <= 1:
        masks = [np.ones(7, float)]
    else:
        masks = [np.ones(7, float)]
        masks += list(hadamard_like_masks(k_virtual - 1, width=7, seed=mask_seed, zero_sum=True))
    masks = np.stack(masks, axis=0)  # [k, 7]

    feats = []
    for img in images_28x28:
        grid = downsample_to_7xM(img, row_bands)  # [row_bands, 7]
        # For each row (7 features), apply all masks and concatenate
        row_feats = []
        for r in range(row_bands):
            # [k,7] * [1,7] -> [k,7], then flatten to [7*k]
            row_feats.append((masks * grid[r:r+1, :]).reshape(-1))
        feats.append(np.concatenate(row_feats, axis=0))
    return np.asarray(feats, dtype=float)

# -----------------------------
# Main (one-layer with optional masks)
# -----------------------------
def main():
    print("="*60)
    print(f"ONE-LAYER CLASSIFIER (downsample → K-masked features → linear)")
    print(f"ROW_BANDS={ROW_BANDS}, K_VIRTUAL={K_VIRTUAL}")
    print("="*60)

    # Load MNIST
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X = mnist.data.values.astype(np.float32) / 255.0
    y = mnist.target.values.astype(int)

    # Balanced subset & reshape
    Xb, yb = make_balanced_subset(X, y, n_per_class=N_PER_CLASS, seed=SEED)
    imgs = Xb.reshape(-1, 28, 28)

    # Build direct-with-masks features
    X_masked = build_direct_with_masks(imgs, row_bands=ROW_BANDS,
                                       k_virtual=K_VIRTUAL, mask_seed=MASK_SEED)
    print(f"Feature shape: {X_masked.shape}  (dim = {7*ROW_BANDS*K_VIRTUAL})")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_masked, yb, test_size=TEST_SIZE, stratify=yb, random_state=SEED
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = {}

    # -------------------------
    # Ridge Classifier (linear)
    # -------------------------
    ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))
    ridge.fit(X_train, y_train)
    y_pred_ridge_tr = ridge.predict(X_train)
    y_pred_ridge_te = ridge.predict(X_test)
    acc_ridge_tr = accuracy_score(y_train, y_pred_ridge_tr)
    acc_ridge_te = accuracy_score(y_test,  y_pred_ridge_te)
    print(f"Ridge Classifier TRAIN accuracy: {acc_ridge_tr:.3f}")
    print(f"Ridge Classifier TEST  accuracy: {acc_ridge_te:.3f}")
    print("\n=== Classification Report (Ridge) ===")
    print(classification_report(y_test, y_pred_ridge_te, zero_division=0))
    print("\n=== Confusion Matrix (Ridge) ===")
    print(confusion_matrix(y_test, y_pred_ridge_te))
    results["ridge"] = {"train_acc": acc_ridge_tr, "test_acc": acc_ridge_te}

    # -------------------------------------------
    # OLS on one-hot (true LinearRegression head)
    # -------------------------------------------
    classes = np.unique(y_train)
    Y_train = np.eye(classes.size)[y_train]
    ols = LinearRegression()
    ols.fit(X_train, Y_train)
    scores_tr = X_train @ ols.coef_.T + ols.intercept_
    scores_te = X_test  @ ols.coef_.T + ols.intercept_
    y_pred_ols_tr = np.argmax(scores_tr, axis=1)
    y_pred_ols_te = np.argmax(scores_te, axis=1)
    acc_ols_tr = accuracy_score(y_train, y_pred_ols_tr)
    acc_ols_te = accuracy_score(y_test,  y_pred_ols_te)
    print(f"\nOLS (LinearRegression) TRAIN accuracy: {acc_ols_tr:.3f}")
    print(f"OLS (LinearRegression) TEST  accuracy: {acc_ols_te:.3f}")
    print("\n=== Classification Report (OLS / LinearRegression) ===")
    print(classification_report(y_test, y_pred_ols_te, zero_division=0))
    print("\n=== Confusion Matrix (OLS / LinearRegression) ===")
    print(confusion_matrix(y_test, y_pred_ols_te))
    results["ols"] = {"train_acc": acc_ols_tr, "test_acc": acc_ols_te}

    # ----------------------
    # Logistic Regression
    # ----------------------
    # logreg = LogisticRegression(max_iter=20000, solver="lbfgs")
    # logreg.fit(X_train, y_train)
    # y_pred_logreg_tr = logreg.predict(X_train)
    # y_pred_logreg_te = logreg.predict(X_test)
    # acc_logreg_tr = accuracy_score(y_train, y_pred_logreg_tr)
    # acc_logreg_te = accuracy_score(y_test,  y_pred_logreg_te)
    # print(f"\nLogistic Regression TRAIN accuracy: {acc_logreg_tr:.3f}")
    # print(f"Logistic Regression TEST  accuracy: {acc_logreg_te:.3f}")
    # print("\n=== Classification Report (Logistic Regression) ===")
    # print(classification_report(y_test, y_pred_logreg_te, zero_division=0))
    # print("\n=== Confusion Matrix (Logistic Regression) ===")
    # print(confusion_matrix(y_test, y_pred_logreg_te))
    # results["logreg"] = {"train_acc": acc_logreg_tr, "test_acc": acc_logreg_te}

    # Summary
    best = max(results.items(), key=lambda kv: kv[1]["test_acc"])
    print("\n" + "-"*60)
    print(f"Best head: {best[0]}  |  test acc = {best[1]['test_acc']:.3f}")
    print("-"*60)

    return results

if __name__ == "__main__":
    main()
