# iris_linear_baselines_scaled.py
import numpy as np
from collections import Counter

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# ==============================
# Config
# ==============================
TEST_SIZE = 0.3
SEED = 42
FEAT_IDX = [0, 1, 2, 3]  # use all 4 features; change to [0,2,3] to test 3-feature configs, etc.
ALPHAS = np.logspace(-3, 3, 13)
CV_SPLITS = 5
CV_RANDOM_STATE = 42


def run_single_split(X, y):
    """One stratified 80/20 split; scale on train only; evaluate ridge + OLS (ELM-compatible)."""
    print("\n" + "=" * 60)
    print("Single 80/20 stratified split (with STANDARDIZATION)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print("y_train counts:", Counter(y_train))
    print("y_test  counts:", Counter(y_test))

    # -------- A) Ridge Classifier (with scaling in a Pipeline) --------
    ridge = make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(alphas=ALPHAS)
    )
    ridge.fit(X_train, y_train)
    y_pred_tr = ridge.predict(X_train)
    y_pred_te = ridge.predict(X_test)
    print(f"\n[Ridge] train acc: {accuracy_score(y_train, y_pred_tr):.3f}")
    print(f"[Ridge]  test acc: {accuracy_score(y_test,  y_pred_te):.3f}")
    print("[Ridge]  report (test):")
    print(classification_report(y_test, y_pred_te, zero_division=0))
    print("[Ridge]  confusion (test):")
    print(confusion_matrix(y_test, y_pred_te))
    ridge_acc = accuracy_score(y_test, y_pred_te)

    # -------- B) OLS on one-hot (LinearRegression head) with scaling --------
    # Build a scaling pipeline just to transform features, then fit OLS on the scaled data.
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    Y_train = np.eye(n_classes)[y_train]
    ols = LinearRegression()
    ols.fit(Xtr, Y_train)

    scores_tr = Xtr @ ols.coef_.T + ols.intercept_
    scores_te = Xte @ ols.coef_.T + ols.intercept_
    y_pred_ols_tr = np.argmax(scores_tr, axis=1)
    y_pred_ols_te = np.argmax(scores_te, axis=1)
    print(f"\n[OLS]   train acc: {accuracy_score(y_train, y_pred_ols_tr):.3f}")
    print(f"[OLS]    test acc: {accuracy_score(y_test,  y_pred_ols_te):.3f}")
    print("[OLS]    report (test):")
    print(classification_report(y_test, y_pred_ols_te, zero_division=0))
    print("[OLS]    confusion (test):")
    print(confusion_matrix(y_test, y_pred_ols_te))
    ols_acc = accuracy_score(y_test, y_pred_ols_te)

    # Summary
    print("\n" + "-" * 60)
    print("Single-split TEST accuracy (scaled features):")
    print(f"Ridge   : {ridge_acc:.3f}")
    print(f"OLS     : {ols_acc:.3f}")
    print("-" * 60)


def run_cross_validation(X, y):
    """5× stratified CV; scale within each fold; report mean±std for each head (ELM-compatible)."""
    print("\n" + "=" * 60)
    print(f"{CV_SPLITS}× Stratified CV (shuffle=True, random_state={CV_RANDOM_STATE}) with STANDARDIZATION")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

    acc_ridge, acc_ols = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        # Ridge pipeline handles scaling internally per fold
        ridge = make_pipeline(
            StandardScaler(),
            RidgeClassifierCV(alphas=ALPHAS)
        ).fit(X[tr], y[tr])
        acc_ridge.append(ridge.score(X[te], y[te]))

        # OLS: fit scaler on train fold, transform both, then fit/predict
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, yte = y[tr], y[te]

        n_classes = len(np.unique(ytr))
        Ytr = np.eye(n_classes)[ytr]
        ols = LinearRegression().fit(Xtr, Ytr)
        ypred_ols = np.argmax(Xte @ ols.coef_.T + ols.intercept_, axis=1)
        acc_ols.append(accuracy_score(yte, ypred_ols))

    def ms(arr):
        return f"{np.mean(arr):.3f} ± {np.std(arr):.3f}"

    print(f"Ridge   : {ms(acc_ridge)}")
    print(f"OLS     : {ms(acc_ols)}")


def main():
    print("=" * 60)
    print("IRIS → linear baselines (Ridge vs OLS) with STANDARDIZATION")
    print(f"Using feature indices: {FEAT_IDX}")
    print("=" * 60)

    iris = load_iris()
    X_full = iris.data.astype(np.float32)   # (150, 4)
    y      = iris.target.astype(int)        # 0,1,2

    X = X_full[:, FEAT_IDX]
    print(f"X shape: {X.shape} (dim = {X.shape[1]})")

    run_single_split(X, y)
    run_cross_validation(X, y)


if __name__ == "__main__":
    main()
