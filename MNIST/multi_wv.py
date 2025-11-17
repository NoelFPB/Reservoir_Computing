import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Lib.scope import RigolDualScopes
from Lib.DualBoard import DualAD5380Controller
from Lib.laser import LaserSource
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeClassifierCV

FEATURE_STORE = os.path.join("MNIST", "3wv.npz")

SCOPE1_CHANNELS = [1, 2, 3, 4]   # first scope (4 channels)
SCOPE2_CHANNELS = [1, 2, 3]      # second scope (3 channels)

INPUT_HEATERS = [28, 29, 30, 31, 32, 33, 34]
ALL_HEATERS = list(range(35))  # Omitting the second part of C
V_MIN, V_MAX = 1.10, 4.90
V_BIAS_INPUT = 3.0
LASER_ADDRESS = "GPIB0::6::INSTR"
ROW_BANDS = 7 # How many 7-wide row bands to use 
K_VIRTUAL = 1          # Still use virtual nodes for feature diversity

READ_AVG = 1             # Fewer averages needed
# Spatial encoding parameters
SPATIAL_GAIN = 0.5     # How strongly pixels drive heaters

# Dataset parameters
N_SAMPLES_PER_DIGIT = 100 # Samples per digit class (500 total for quick demo)
TEST_FRACTION = 0.3      # % for testing


WAVELENGTHS = [1548.0, 1550.0, 1552.0]  # or however many you want

#
def select_images_for_missing(X_pool, y_pool, missing_per_class, seed=42):
    """
    From a candidate pool (X_pool, y_pool), pick exactly the number of
    additional samples required per class (given by missing_per_class).
    Returns X_new, y_new.
    """
    rng = np.random.default_rng(seed)

    # We'll iterate in random order to avoid always picking the same ones
    indices = rng.permutation(len(y_pool))

    missing = missing_per_class.copy()
    X_new_list, y_new_list = [], []

    for idx in indices:
        label = y_pool[idx]
        if missing[label] > 0:
            X_new_list.append(X_pool[idx])
            y_new_list.append(label)
            missing[label] -= 1

            if np.all(missing <= 0):
                break

    if not X_new_list:
        print("[WARN] select_images_for_missing: could not find any new samples.")
        return np.empty((0, X_pool.shape[1])), np.empty((0,), dtype=int)

    X_new = np.vstack(X_new_list)
    y_new = np.array(y_new_list, dtype=int)
    print(f"[SELECT] Selected {len(y_new)} new images (per-class missing now: {missing})")
    return X_new, y_new

def load_latest_multi_lambda(base_dir="MNIST/multi_wavelength"):
    base_path = Path(base_dir)
    files = list(base_path.glob("multi_lambda_*.npz"))
    if not files:
        raise FileNotFoundError(f"No multi_lambda_*.npz files found in {base_dir}")
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"[FASTPATH] Using existing multi-λ dataset: {latest}")
    return load_multi_wavelength_features(str(latest))

def load_multi_wavelength_features(path):
    """
    Load a multi-wavelength NPZ dataset saved by save_multi_wavelength_features().
    Returns dict with:
        X_stack:   (N, L, D)
        X_concat:  (N, L*D)
        y:         (N,)
        wavelengths: (L,)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    d = np.load(path)

    data = {
        "X_stack": d["X_stack"],
        "X_concat": d["X_concat"],
        "y": d["y"],
        "wavelengths": d["wavelengths"],
        "path": path,
    }

    print(f"[LOAD] Loaded multi-λ dataset from:\n  {path}")
    print(f"       X_stack shape   : {data['X_stack'].shape} (N, L, D)")
    print(f"       X_concat shape  : {data['X_concat'].shape}")
    print(f"       wavelengths (nm): {data['wavelengths']}")
    return data

def save_multi_wavelength_features(
        X_list, y, wavelengths, base_dir="MNIST/multi_wavelength"
    ):
    """
    X_list: list of feature matrices [X_λ0, X_λ1, ...], each shape (N, D)
    y:      labels, shape (N,)
    wavelengths: list/array of wavelengths (floats)
    """
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(base_dir, f"multi_lambda_{ts}.npz")

    X_concat = np.hstack(X_list)
    X_stack  = np.stack(X_list, axis=1)  # (N, L, D)

    np.savez_compressed(
        filename,
        X_stack=X_stack.astype(np.float32),
        X_concat=X_concat.astype(np.float32),
        y=y.astype(np.int64),
        wavelengths=np.array(wavelengths, dtype=np.float32),
    )

    print(f"[SAVE] Multi-wavelength features saved to:\n  {os.path.abspath(filename)}")
    return filename

def load_feature_store(path=FEATURE_STORE):
    if os.path.exists(path):
        d = np.load(path)
        return d["X"], d["y"]
    return None, None

def save_feature_store(X, y, path=FEATURE_STORE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))
    print(f"[FeatureStore] Saved {len(y)} samples to {path}")

def append_feature_store(X_new, y_new, path=FEATURE_STORE):
    X_old, y_old = load_feature_store(path)
    if X_old is None:
        X_all, y_all = X_new, y_new
    else:
        X_all = np.concatenate([X_old, X_new], axis=0)
        y_all = np.concatenate([y_old, y_new], axis=0)
    save_feature_store(X_all, y_all, path)
    return X_all, y_all

def per_class_counts(y, n_classes=10):
    counts = np.zeros(n_classes, dtype=int)
    if y is not None:
        for c in range(n_classes):
            counts[c] = int(np.sum(y == c))
    return counts

def pick_balanced_subset(X, y, n_per_class, seed=42):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(10):
        idx = np.where(y == c)[0]
        take = rng.choice(idx, size=n_per_class, replace=False)
        xs.append(X[take])
        ys.append(y[take])
    return np.vstack(xs), np.concatenate(ys)
#
      
def hadamard_like_masks(n_masks, width=7, seed=0):
    rng = np.random.default_rng(seed)
    M = []
    while len(M) < n_masks:
        v = rng.choice([-1.0, 1.0], size=width)
        # accept only if nearly orthogonal to existing ones
        if all(abs(np.dot(v, m)/width) < 0.2 for m in M):
            M.append(v)
    return np.array(M)

def downsample_to_7xM(img2d: np.ndarray, M: int) -> np.ndarray:
    assert img2d.shape == (28, 28)
    # columns: 28 -> 7 (keep horizontal detail for each heater)
    col_reduced = img2d.reshape(28, 7, 4).mean(axis=2)  # (28, 7)
    # rows: 28 -> M bands
    bands = np.array_split(col_reduced, M, axis=0)     # list of (rows_i, 7)
    out = np.stack([b.mean(axis=0) for b in bands], axis=0)  # (M, 7)
    return out

def load_mnist_pool(max_per_class=400):
    """Return a larger candidate pool (downsampled to 7xROW_BANDS) to fill 'missing' needs."""
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.values / 255.0, mnist.target.values.astype(int)
    X_resized = []
    for img in X:
        img_2d = img.reshape(28, 28)
        grid7xM = downsample_to_7xM(img_2d, ROW_BANDS)
        X_resized.append(grid7xM.flatten())
    X_resized = np.array(X_resized)

    # Create a balanced pool up to max_per_class
    X_pool, y_pool = create_balanced_subset(X_resized, y, max_per_class)
    return X_pool, y_pool
    
        
def create_balanced_subset(X, y, n_per_class):
    """Create balanced subset with n_per_class samples of each digit."""
    X_subset = []
    y_subset = []
    
    for digit in range(10):  # Digits 0-9
        digit_indices = np.where(y == digit)[0]
        selected_indices = np.random.choice(
            digit_indices, 
            size=min(n_per_class, len(digit_indices)), 
            replace=False
        )
        
        X_subset.extend(X[selected_indices])
        y_subset.extend([digit] * len(selected_indices))
    
    # Shuffle the combined dataset
    indices = np.random.permutation(len(X_subset))
    X_subset = np.array(X_subset)[indices]
    y_subset = np.array(y_subset)[indices]
    
    return X_subset, y_subset

def measure_dataset_at_wavelength(reservoir, X_images, y_labels,
                                  wavelength_nm, phase_tag="MEASURE"):
    """
    Set the laser wavelength, turn it on, run the dataset through the reservoir,
    then turn it off. Returns (X_features, y_labels) in the same order as input.
    """
    print("="*60)
    print(f"[{phase_tag}] Measuring at {wavelength_nm} nm")
    print("="*60)

    # Tune laser
    reservoir.laser.set_wavelength(wavelength_nm, settle=0.5)
    reservoir.laser.turn_on(settle=12)
    

    # IMPORTANT: target_per_class=None → process_dataset will not skip anything
    X_feat, y_feat = reservoir.process_dataset(
        X_images, y_labels,
        phase_name=f"{phase_tag}_{wavelength_nm}nm",
        existing_counts=None,
        target_per_class=None
    )

    reservoir.laser.turn_off(settle=0.5)
    return X_feat, y_feat


# ==========================
# CLASSES
# ==========================
class PhotonicReservoir:
    """Base photonic reservoir class."""
    def __init__(self, input_heaters, all_heaters):
        self.input_heaters = list(input_heaters)
        self.internal_heaters = [h for h in all_heaters if h not in self.input_heaters]
        self.scope = RigolDualScopes(SCOPE1_CHANNELS, SCOPE2_CHANNELS,
                                     serial_scope1='HDO1B244000779')
        self.bus = DualAD5380Controller()

        # NEW: laser handle
        self.laser = LaserSource(LASER_ADDRESS, timeout_ms=5000,
                                 write_termination="", read_termination="")

        # --- your existing mesh_bias, input_bias, baseline code here ---
        rng = np.random.default_rng(40)
        self.mesh_bias = {
            h: float(rng.uniform(0.5, 4.5))
            for h in self.internal_heaters
        }
        non_linear = { 
        }
        non_linear_int = {int(k): float(v) for k, v in non_linear.items()}
        #self.mesh_bias = {h: non_linear_int[h] for h in self.internal_heaters}
        self.input_bias = {h: V_BIAS_INPUT for h in range(28, 35)}

        baseline = ({**self.mesh_bias, **self.input_bias})
        chs = sorted(baseline.keys())
        vs = [baseline[h] for h in chs]
        self.bus.set(chs, vs)

        print(self.input_bias)
        print(self.mesh_bias)

    def close(self):
        # turn laser off before closing
        try:
            self.laser.turn_off(settle=0.0)
        except Exception:
            pass
        self.scope.close()
        self.laser.close()

# ==========================
# PHOTONIC RESERVOIR
# ==========================

class PhotonicReservoirMNIST(PhotonicReservoir):
    def __init__(self, input_heaters, all_heaters):
        super().__init__(input_heaters, all_heaters)

    def process_spatial_pattern(self, image_pixels):
        K_req   = int(K_VIRTUAL)
        readavg = int(READ_AVG)
        vmin    = float(V_MIN)
        vmax    = float(V_MAX)
        gain    = float(SPATIAL_GAIN)
        v_bias  = float(V_BIAS_INPUT)
        heaters     = self.input_heaters
        chunk_size  = len(heaters)                # 7

        x = np.asarray(image_pixels, float).ravel()
        total_pixels = (x.size // chunk_size) * chunk_size
        x = x[:total_pixels]
        num_chunks = total_pixels // chunk_size

        half_swing = 0.5 * (vmax - vmin)
        x_centered = (x - 0.5) * 2.0              # [-1, 1]

        # ---- build mask matrix once (unified path) ----
        # K_eff = 1 uses [ones] -> identical to "no mask" behavior (fast)
        if K_req <= 1:
            mask_matrix = np.ones((1, chunk_size), dtype=float)
        else:
            need = K_req - 1
            assert len(self.mask) >= need, \
                f"Need {need} micro masks, have {len(self.mask)}"
            mask_matrix = np.vstack([
                np.ones((1, chunk_size), dtype=float),             # baseline mask
                np.asarray(self.mask[:need], dtype=float)   # +/-1 masks
            ])

        send = self.bus.set
        read_many = self.scope.read_many

        features = []

        for i in range(num_chunks):
            sl = slice(i*chunk_size, (i+1)*chunk_size)
            base = gain * half_swing * x_centered[sl]            # shape (7,)
            # iterate masks (K_eff times; when K_eff==1 this is minimal)
            for m in mask_matrix:
                v = v_bias + base * m
                v = np.clip(v, vmin, vmax)
                #print(v)
                #print("v", np.round(v, 2))
                send(heaters, v.tolist())                   
                pd = read_many(avg=readavg)
                #print(pd)
                features.append(pd)

        return np.asarray(features, float).ravel()

    def process_dataset(self, X_images, y_labels, phase_name="PROCESSING",
                    existing_counts=None, target_per_class=None):

        print(f"[{phase_name}] Candidate images: {len(X_images)}")
        need = int(target_per_class) if target_per_class is not None else None
        have = per_class_counts(y=None) if existing_counts is None else existing_counts.copy()

        # If nothing missing, bail out immediately
        if need is not None and np.all(have >= need):
            print(f"[{phase_name}] Already have >={need} per class. No measurement needed.")
            return np.empty((0, )), np.empty((0, ), dtype=int)

        X_new, y_new = [], []
        missing = None if need is None else (np.maximum(need - have, 0))

        for i, (image, label) in enumerate(zip(X_images, y_labels)):
            if (i + 1) % 25 == 0:
                print(f"[{phase_name}] Processed {i+1}/{len(X_images)} images")
            if need is not None:
                # If this class is already satisfied, skip without touching hardware
                if missing[label] <= 0:
                    continue
                # If ALL satisfied, stop
                if np.all(missing <= 0):
                    break

            # Drive hardware for this one
            try:
                features = self.process_spatial_pattern(image)
                if not np.any(np.isnan(features)):
                    X_new.append(features)
                    y_new.append(label)
                    if need is not None:
                        have[label] += 1
                        missing[label] -= 1
                else:
                    print(f"[WARNING] NaN features at candidate {i}; skipped.")
            except Exception as e:
                print(f"[ERROR] Failed at candidate {i}: {e}")

        if len(X_new) == 0:
            print(f"[{phase_name}] No new samples measured.")
            return np.empty((0, )), np.empty((0, ), dtype=int)

        X_new = np.asarray(X_new, float)
        y_new = np.asarray(y_new, int)
        print(f"[{phase_name}] Measured new: {len(y_new)} (per-class now: {have})")
        return X_new, y_new
# ==========================
# CLASSIFICATION TRAINING
# ==========================


def train_mnist_classifier(X_features, y_labels, *, seed=42):
    """
    Trains THREE one-layer heads on fixed features (ELM style):
      1) LinearRegression (multi-output on one-hot; predict via argmax)
      2) RidgeClassifierCV (direct linear classifier on labels)
      3) LogisticRegression (multinomial linear classifier)
    """
    TEST_FRACTION = 0.2

    print("\n[ELM] Training one-layer heads on fixed features...")
    print(f"[ELM] Input fixed feature dim: {X_features.shape[1]}")

    # 1) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels,
        test_size=TEST_FRACTION,
        stratify=y_labels,
        random_state=seed
    )
    print(f"[ELM] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2) Standardize features (shared)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 3) One-hot for the regression head
    classes = np.unique(y_labels)
    n_classes = classes.size
    Y_train = np.eye(n_classes)[y_train]

    models = {}
    results = {}

    # ---- A) Linear Regression (multi-output OLS on one-hot) ----
    linreg = LinearRegression(fit_intercept=True)
    linreg.fit(X_train_s, Y_train)

    scores_tr = X_train_s @ linreg.coef_.T + linreg.intercept_
    y_pred_tr = np.argmax(scores_tr, axis=1)
    acc_tr = accuracy_score(y_train, y_pred_tr)

    scores_te = X_test_s @ linreg.coef_.T + linreg.intercept_
    y_pred_te = np.argmax(scores_te, axis=1)
    acc_te = accuracy_score(y_test, y_pred_te)

    print(f"\n[LINREG] Train acc: {acc_tr:.3f} | Test acc: {acc_te:.3f}")
    print("[LINREG] Classification report (test):")
    print(classification_report(y_test, y_pred_te, zero_division=0))
    print("[LINREG] Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred_te))

    models["linreg"] = linreg
    results["linreg"] = {
        "train_accuracy": acc_tr,
        "test_accuracy": acc_te,
        "y_pred": y_pred_te,
        "n_features": X_features.shape[1],
        "n_classes": n_classes,
    }

    # ---- B) RidgeClassifierCV (direct linear classifier with L2) ----
    ridge_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))
    ridge_clf.fit(X_train_s, y_train)

    y_pred_tr2 = ridge_clf.predict(X_train_s)
    y_pred_te2 = ridge_clf.predict(X_test_s)
    acc_tr2 = accuracy_score(y_train, y_pred_tr2)
    acc_te2 = accuracy_score(y_test, y_pred_te2)

    print(f"\n[RIDGE-CLF] Train acc: {acc_tr2:.3f} | Test acc: {acc_te2:.3f}")
    print("[RIDGE-CLF] Classification report (test):")
    print(classification_report(y_test, y_pred_te2, zero_division=0))
    print("[RIDGE-CLF] Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred_te2))
    print(f"[RIDGE-CLF] Chosen alpha: {ridge_clf.alpha_}")

    models["ridge_clf"] = ridge_clf
    results["ridge_clf"] = {
        "train_accuracy": acc_tr2,
        "test_accuracy": acc_te2,
        "y_pred": y_pred_te2,
        "alpha": ridge_clf.alpha_,
        "n_features": X_features.shape[1],
        "n_classes": n_classes,
    }

    # store scaler for later inference
    results["scaler"] = scaler

    return models, results, (X_test, y_test)
# ==========================
# VISUALIZATION AND ANALYSIS
# ==========================

def visualize_results(X_images, y_labels, classifier, X_test, y_test, results, *,
                      meta=None, total_seconds=None, run_tag=None):
    
    # Prefer a specific model if present; fall back to best available
    preferred_order = ["logreg", "ridge_clf", "linreg"]
    key = None
    for name in preferred_order:
        if isinstance(results.get(name), dict) and "y_pred" in results[name]:
            key = name
            break
    if key is None:
        # fallback: pick the entry with y_pred and highest test_accuracy
        candidates = [(k, v) for k, v in results.items()
                      if isinstance(v, dict) and "y_pred" in v]
        if not candidates:
            raise ValueError("No results with 'y_pred' found for visualization.")
        key = max(candidates, key=lambda kv: kv[1].get("test_accuracy", float("-inf")))[0]

    y_pred = np.asarray(results[key]["y_pred"])

    # ensure output folder exists
    save_dir=os.path.join("MNIST", "figures")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))

    # --- Confusion Matrix ---
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest')  # default colormap
    plt.title(f'Confusion Matrix ({key})')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # --- Per-digit accuracy ---
    plt.subplot(1, 2, 2)
    digit_accuracies = []
    for digit in range(10):
        mask = (y_test == digit)
        if np.any(mask):
            acc = accuracy_score(y_test[mask], y_pred[mask])
        else:
            acc = 0.0
        digit_accuracies.append(acc)
    plt.bar(range(10), digit_accuracies)
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.title(f'Per-Digit Accuracy ({key})')
    plt.xticks(range(10))

    # --- Title metadata (robust best model selection) ---
    meta = meta or {}
    # choose best by test_accuracy among dict entries that have it
    candidate_items = [(k, v) for k, v in results.items()
                       if isinstance(v, dict) and "test_accuracy" in v]
    if candidate_items:
        best_name, best_info = max(candidate_items, key=lambda kv: kv[1]["test_accuracy"])
        best_acc = best_info["test_accuracy"]
    else:
        best_name, best_acc = key, float('nan')

    K   = meta.get("K_VIRTUAL", "N/A")
    bands = meta.get("ROW_BANDS", "N/A")
    nspd  = meta.get("N_SAMPLES_PER_DIGIT", "N/A")
    gain  = meta.get("SPATIAL_GAIN", "N/A")
    read_avg = meta.get("READ_AVG", "N/A")
    time_str = f"{total_seconds:.2f}s" if isinstance(total_seconds, (int, float)) else "N/A"

    meta_line_1 = f"Best: {best_name}  |  Test Acc: {best_acc:.3f}  |  Time: {time_str}"
    meta_line_2 = f"K={K}, ROW_BANDS={bands}, N_SAMPLES_PER_DIGIT={nspd}, SPATIAL_GAIN={gain}, READ_AVG={read_avg}"
    plt.suptitle(meta_line_1 + "\n" + meta_line_2, y=1.05, fontsize=11)
    plt.tight_layout()

    # --- Save figure ---
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    fname_base = f"results{tag}_{ts}"
    png_path = os.path.join(save_dir, fname_base + ".png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"[Saved] {png_path}")


def analyze_feature_importance(classifier, feature_names=None):
    """
    Analyze which features are most important for classification.
    """
    if hasattr(classifier, 'coef_'):
        importance = np.abs(classifier.coef_).mean(axis=0)
        
        print("\nFeature Importance Analysis:")
        print(f"Most important features (top 10):")
        
        top_indices = np.argsort(importance)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
            print(f"  {i+1:2d}. {feature_name}: {importance[idx]:.4f}")

# ==========================
# MAIN EXECUTION
# ==========================
def main_mnist_dual_wavelength():
    """
    Multi-wavelength MNIST experiment (3 wavelengths in WAVELENGTHS):

    - If a multi-λ dataset exists and has enough samples per class, reuse it.
    - Otherwise, measure a fresh balanced dataset at all wavelengths.
    """
    print("="*60)
    print("MNIST CLASSIFICATION – MULTI WAVELENGTH")
    print("="*60)

    t0 = time.perf_counter()
    reservoir = None

    try:
        # ----------------------------------------
        # TRY TO LOAD EXISTING MULTI-λ DATASET
        # ----------------------------------------
        try:
            data = load_latest_multi_lambda(base_dir="MNIST/dual_wavelength")
            X_stack = data["X_stack"]        # (N, L, D)
            X_feat_concat = data["X_concat"] # (N, L*D)
            y_all = data["y"]               # (N,)
            wavelengths = data["wavelengths"]

            counts = per_class_counts(y_all, n_classes=10)
            print(f"[FASTPATH] Existing per-class counts: {counts} (target={N_SAMPLES_PER_DIGIT})")
            print(f"[FASTPATH] Existing wavelengths (nm): {wavelengths}")

            if np.all(counts >= N_SAMPLES_PER_DIGIT) and len(wavelengths) == len(WAVELENGTHS):
                print("[FASTPATH] Existing multi-λ dataset satisfies target. Skipping measurement.")
            else:
                raise FileNotFoundError(
                    f"Existing multi-λ dataset does not satisfy target or λ count mismatch "
                    f"(have counts={counts}, wavelengths={wavelengths}, target={N_SAMPLES_PER_DIGIT}, "
                    f"expected L={len(WAVELENGTHS)})."
                )

        except FileNotFoundError as e:
            # ----------------------------------------
            # NO SUITABLE FILE → FRESH MEASUREMENT
            # ----------------------------------------
            print(f"[INFO] {e}")
            print("[INFO] Starting full hardware measurement for multi-λ run...")

            print("[SETUP] Loading MNIST pool...")
            X_pool, y_pool = load_mnist_pool(max_per_class=N_SAMPLES_PER_DIGIT)

            print("[SETUP] Creating balanced subset...")
            X_imgs, y_imgs = create_balanced_subset(
                X_pool, y_pool, n_per_class=N_SAMPLES_PER_DIGIT
            )
            print(f"[SETUP] Using {len(y_imgs)} images total for measurement.")

            reservoir = PhotonicReservoirMNIST(INPUT_HEATERS, ALL_HEATERS)

            X_feats = []   # list of (N, D) arrays, one per wavelength
            y_ref   = None

            for i, wl in enumerate(WAVELENGTHS):
                X_i, y_i = measure_dataset_at_wavelength(
                    reservoir, X_imgs, y_imgs, wl, phase_tag=f"LAMBDA{i+1}"
                )
                X_feats.append(X_i)

                if y_ref is None:
                    y_ref = y_i
                else:
                    if not np.array_equal(y_ref, y_i):
                        raise RuntimeError(f"Label mismatch between wavelength runs (index {i})!")

            y_all = y_ref
            print("[MEASURE] Feature shapes per λ:")
            for wl, X_i in zip(WAVELENGTHS, X_feats):
                print(f"  λ={wl} nm: {X_i.shape}")

            # Build X_stack and concatenated features
            X_stack = np.stack(X_feats, axis=1)      # (N, L, D)
            X_feat_concat = np.hstack(X_feats)       # (N, L*D)
            print(f"[COMBINE] X_stack shape  : {X_stack.shape}  (N, L, D)")
            print(f"[COMBINE] X_concat shape : {X_feat_concat.shape}")

            # Save the multi-λ dataset
            save_path = save_multi_wavelength_features(
                X_feats, y_all, WAVELENGTHS,
                base_dir="MNIST/dual_wavelength"
            )
            print(f"[INFO] Multi-λ dataset saved at: {save_path}")

        # ----------------------------------------
        # COMMON PART: TRAINING & VISUALIZATION
        # ----------------------------------------
        N, L, D = X_stack.shape
        print(f"[TRAIN] N={N}, L={L}, D={D}")

        # Train one classifier per wavelength
        per_lambda_results = []
        for i, wl in enumerate(WAVELENGTHS):
            X_i = X_stack[:, i, :]  # (N, D)
            print(f"\n=== TRAINING: λ={wl} nm only ===")
            models_i, results_i, test_data_i = train_mnist_classifier(X_i, y_all)
            per_lambda_results.append((wl, models_i, results_i, test_data_i))

        # Train on concatenated features
        print("\n=== TRAINING: all λ concatenated ===")
        models_cat, results_cat, test_data_cat = train_mnist_classifier(X_feat_concat, y_all)

        total_time = time.perf_counter() - t0

        meta = {
            "K_VIRTUAL": K_VIRTUAL,
            "ROW_BANDS": ROW_BANDS,
            "N_SAMPLES_PER_DIGIT": N_SAMPLES_PER_DIGIT,
            "SPATIAL_GAIN": SPATIAL_GAIN,
            "READ_AVG": READ_AVG,
        }

        visualize_results(
            None, None,
            models_cat,
            *test_data_cat,
            results_cat,
            meta=meta,
            total_seconds=total_time,
            run_tag="mnist_multi_lambda"
        )

        # Summary helper
        def best_acc(res_dict):
            metrics = [(k, v["test_accuracy"]) for k, v in res_dict.items()
                       if isinstance(v, dict) and "test_accuracy" in v]
            if not metrics:
                return None, 0.0
            return max(metrics, key=lambda kv: kv[1])

        print("\n========== ACCURACY SUMMARY ==========")
        for (wl, models_i, results_i, test_data_i) in per_lambda_results:
            name_i, acc_i = best_acc(results_i)
            print(f"λ={wl:.1f} nm : {name_i} → {acc_i:.3f}")

        nameC, accC = best_acc(results_cat)
        print(f"ALL λ concatenated : {nameC} → {accC:.3f}")
        print("======================================")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error in multi-wavelength run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if reservoir is not None:
            try:
                reservoir.close()
            except Exception:
                pass


if __name__ == "__main__":  
    main_mnist_dual_wavelength()
