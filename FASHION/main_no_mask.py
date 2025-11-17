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
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeClassifierCV, LogisticRegression

FEATURE_STORE = os.path.join("FASHION", "R7_G08_0bias.npz")

FASHION_CLASSES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                   'Sandal','Shirt','Sneaker','Bag','Ankle boot']

SCOPE1_CHANNELS = [1, 2, 3, 4]   # first scope (4 channels)
SCOPE2_CHANNELS = [1, 2, 3]      # second scope (3 channels)

INPUT_HEATERS = [28, 29, 30, 31, 32, 33, 34]
ALL_HEATERS = list(range(35))  # Omitting the second part of C
V_MIN, V_MAX = 2.0, 4.50

ROW_BANDS = 5 # How many 7-wide row bands to use 

READ_AVG = 1             # Fewer averages needed
# Spatial encoding parameters
SPATIAL_GAIN = 0.8     # How strongly pixels drive heaters Now should be less than 0.64

# Dataset parameters
N_SAMPLES_PER_DIGIT = 100 # Samples per digit class (500 total for quick demo)z
TEST_FRACTION = 0.2      # 20% for testing

#
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
    #save_feature_store(X_all, y_all, path)
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
      
def downsample_to_7xM(img2d: np.ndarray, M: int) -> np.ndarray:
    assert img2d.shape == (28, 28)
    # columns: 28 -> 7 (keep horizontal detail for each heater)
    col_reduced = img2d.reshape(28, 7, 4).mean(axis=2)  # (28, 7)
    # rows: 28 -> M bands
    bands = np.array_split(col_reduced, M, axis=0)     # list of (rows_i, 7)
    out = np.stack([b.mean(axis=0) for b in bands], axis=0)  # (M, 7)
    return out

def load_fashion_mnist_pool(max_per_class=400):
    """
    Return a larger candidate pool (downsampled to 7xROW_BANDS)
    to fill 'missing' needs, but using Fashion-MNIST instead of MNIST.
    """
    print("[Data] Loading Fashion-MNIST pool...")
    fm = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X, y = fm.data / 255.0, fm.target.astype(int)

    X_resized = []
    for img in X:
        img_2d = img.reshape(28, 28)
        grid7xM = downsample_to_7xM(img_2d, ROW_BANDS)   # (ROW_BANDS × 7)
        X_resized.append(grid7xM.flatten())

    X_resized = np.asarray(X_resized)

    # Balanced candidate pool
    X_pool, y_pool = create_balanced_subset(X_resized, y, max_per_class)
    print(f"[Data] Fashion-MNIST pool ready: {len(X_pool)} samples "
          f"({max_per_class} per class).")
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

# ==========================
# CLASSES
# ==========================
class PhotonicReservoir:
    """Base photonic reservoir class."""
    def __init__(self, input_heaters, all_heaters):
        self.input_heaters = list(input_heaters)
        self.internal_heaters = [h for h in all_heaters if h not in self.input_heaters]
        self.scope = RigolDualScopes(SCOPE1_CHANNELS, SCOPE2_CHANNELS, serial_scope1='HDO1B244000779')
        self.bus = DualAD5380Controller()

        # Fixed random mesh bias

        rng = np.random.default_rng(42)
        self.mesh_bias = {
            h: float(rng.uniform(V_MIN, V_MAX))
            for h in self.internal_heaters
        }

        non_linear={
            "0": 2.4307063206211885,
            "1": 2.4051826233579345,
            "2": 2.1441468406667985,
            "3": 2.6157552636055854,
            "4": 2.9299635513432296,
            "5": 1.4245174875562898,
            "6": 3.5853399046845267,
            "7": 3.3629933051327305,
            "8": 3.8881657839003725,
            "9": 2.140515609371937,
            "10": 2.6865694960639592,
            "11": 3.0791713693298854,
            "12": 4.0240410871001915,
            "13": 3.806837579834224,
            "14": 1.2,
            "15": 4.235503213304205,
            "16": 3.5601539278775727,
            "17": 3.3212197141805504,
            "18": 4.035341949357424,
            "19": 2.287130742651246,
            "20": 3.7000766515900514,
            "21": 3.4006452314023567,
            "22": 3.8810435836174895,
            "23": 1.2004052838182613,
            "24": 2.0878409385025143,
            "25": 3.5562217701588175,
            "26": 2.7177956532175527,
            "27": 1.391873431242149
        }

        
        inputs_bias = [3.25] * len(self.input_heaters)
        self.bus.set(self.input_heaters, inputs_bias)

        print(inputs_bias)

        baseline = non_linear
        chs = baseline.keys()
        vs = baseline.values()
        print(chs)
        print(vs)
        self.bus.set(chs, vs)


    def close(self):
        self.scope.close()

# ==========================
# PHOTONIC RESERVOIR
# ==========================

class PhotonicReservoirMNIST(PhotonicReservoir):
    def __init__(self, input_heaters, all_heaters):
        super().__init__(input_heaters, all_heaters)
        
    def process_spatial_pattern(self, image_pixels):
        readavg = int(READ_AVG)
        vmin    = float(V_MIN)
        vmax    = float(V_MAX)
        gain    = float(SPATIAL_GAIN)
        heaters     = self.input_heaters
        chunk_size  = len(heaters)                # 7

        x = np.asarray(image_pixels, float).ravel()
        total_pixels = (x.size // chunk_size) * chunk_size
        x = x[:total_pixels]
        num_chunks = total_pixels // chunk_size

        send = self.bus.set
        read_many = self.scope.read_many

        features = []
        for i in range(num_chunks):
            sl = slice(i*chunk_size, (i+1)*chunk_size)
 
            v = 0.5 * (vmax + vmin) + gain * (x[sl] - 0.5) * (vmax - vmin)
            v = np.clip(v, vmin, vmax)
            #print(v)
            send(heaters, v.tolist())            
            pd = read_many(avg=readavg)
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

    # ---- C) Logistic Regression (multinomial) ----
    # logreg = LogisticRegression(max_iter=20000, solver="lbfgs", C=0.3)
    # logreg.fit(X_train_s, y_train)

    # y_pred_tr3 = logreg.predict(X_train_s)
    # y_pred_te3 = logreg.predict(X_test_s)
    # acc_tr3 = accuracy_score(y_train, y_pred_tr3)
    # acc_te3 = accuracy_score(y_test, y_pred_te3)

    # print(f"\n[LOGREG] Train acc: {acc_tr3:.3f} | Test acc: {acc_te3:.3f}")
    # print("[LOGREG] Classification report (test):")
    # print(classification_report(y_test, y_pred_te3, zero_division=0))
    # print("[LOGREG] Confusion matrix (test):")
    # print(confusion_matrix(y_test, y_pred_te3))

    # models["logreg"] = logreg
    # results["logreg"] = {
    #     "train_accuracy": acc_tr3,
    #     "test_accuracy": acc_te3,
    #     "y_pred": y_pred_te3,
    #     "n_features": X_features.shape[1],
    #     "n_classes": n_classes,
    # }

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
    save_dir=os.path.join("FASHION", "figures")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))

    # --- Confusion Matrix ---
    labels = FASHION_CLASSES  # for Fashion-MNIST; keep range(10) for digits

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion Matrix ({key})')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(ticks=range(10), labels=labels, rotation=45, ha='right', fontsize=8)
    plt.yticks(ticks=range(10), labels=labels, fontsize=8)

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

    bands = meta.get("ROW_BANDS", "N/A")
    nspd  = meta.get("N_SAMPLES_PER_DIGIT", "N/A")
    gain  = meta.get("SPATIAL_GAIN", "N/A")
    read_avg = meta.get("READ_AVG", "N/A")
    time_str = f"{total_seconds:.2f}s" if isinstance(total_seconds, (int, float)) else "N/A"

    meta_line_1 = f"Best: {best_name}  |  Test Acc: {best_acc:.3f}  |  Time: {time_str}"
    meta_line_2 = f"ROW_BANDS={bands}, N_SAMPLES_PER_DIGIT={nspd}, SPATIAL_GAIN={gain}, READ_AVG={read_avg}"
    plt.suptitle(meta_line_1 + "\n" + meta_line_2, y=1.05, fontsize=11)
    plt.tight_layout()

    # --- Save figure ---
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    fname_base = f"mnist_photonic_results{tag}_{ts}"
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
def main_mnist():
    print("="*60)
    print("MNIST CLASSIFICATION")
    print("="*60)
    t0 = time.perf_counter()
    reservoir = None
    try:
        # ---------- FAST PATH: skip hardware if cache already satisfies target ----------
        X_cached, y_cached = load_feature_store(FEATURE_STORE)
        target = N_SAMPLES_PER_DIGIT
        if X_cached is not None:
            counts = per_class_counts(y_cached, n_classes=10)
            if np.all(counts >= target):
                print(f"[FastPath] Cache already has >= {target} per class. Skipping measurement.")
                # Train using exactly 'target' per class
                X_train_pool, y_train_pool = pick_balanced_subset(X_cached, y_cached, target, seed=42)
                models, results, test_data = train_mnist_classifier(X_train_pool, y_train_pool)
                scaler = results.pop("scaler", None)

                # (optional) visualize & save model exactly as you already do
                meta = {
                    "ROW_BANDS": ROW_BANDS,
                    "N_SAMPLES_PER_DIGIT": N_SAMPLES_PER_DIGIT,
                    "SPATIAL_GAIN": SPATIAL_GAIN, "READ_AVG": READ_AVG,
                }
                total_time = time.perf_counter() - t0
                visualize_results(None, None, models, *test_data, results,
                                  meta=meta,
                                  total_seconds=total_time, run_tag="mnist_photonic_cached")
                # Save artifacts (reusing your existing save block)
                os.makedirs("models", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join("MNIST","models", f"mnist_photonic_classifier_{timestamp}.pkl")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump({'models': models, 'results': results,
                                 'config': {'INPUT_HEATERS': INPUT_HEATERS,
                                            'SPATIAL_GAIN': SPATIAL_GAIN,
                                            }}, f)
                print(f"Classifier saved to '{save_path}'")
                return
        # ---------- END FAST PATH ----------
        else:
            counts = np.zeros(10, dtype=int)
        # Get a big candidate pool; we will only measure what’s missing
        X_pool, y_pool = load_fashion_mnist_pool(max_per_class=target)

        reservoir = PhotonicReservoirMNIST(INPUT_HEATERS, ALL_HEATERS)

        # Measure ONLY the missing samples per class; stop when each class hits 'target'
        X_new, y_new = reservoir.process_dataset(
            X_pool, y_pool, "TRAINING",
            existing_counts=counts,
            target_per_class=target
        )

        if reservoir is not None:
            reservoir.close(); reservoir = None

        # If nothing new was needed, fine; otherwise append
        if X_new.size > 0:
            X_all, y_all = append_feature_store(X_new, y_new, FEATURE_STORE)
        else:
            X_all, y_all = (X_cached, y_cached)

        # Now we definitely have at least what we had before,
        # and likely exactly 'target' per class. Train on a balanced slice:
        counts_after = per_class_counts(y_all, 10)
        if np.any(counts_after < target):
            print(f"[WARN] Still short for some classes: {counts_after} (need {target}).")
            # You can bail out or train on what you have; here we bail for clarity:
            return

        X_for_train, y_for_train = pick_balanced_subset(X_all, y_all, target, seed=42)
        models, results, test_data = train_mnist_classifier(X_for_train, y_for_train)
        scaler = results.pop("scaler", None)

        # Analysis
        clf_for_importance = models.get("ridge_clf", models.get("linreg"))
        if clf_for_importance is not None:
            analyze_feature_importance(clf_for_importance)

        total_time = time.perf_counter() - t0

        meta = {
            "ROW_BANDS": ROW_BANDS,
            "N_SAMPLES_PER_DIGIT": N_SAMPLES_PER_DIGIT,
            "SPATIAL_GAIN": SPATIAL_GAIN,
            "READ_AVG": READ_AVG,
        }

        if 'X_all' in locals() and X_all is not None and len(X_all) > 0:
            X_vis, y_vis = X_all, y_all
        elif 'X_cached' in locals() and X_cached is not None and len(X_cached) > 0:
            X_vis, y_vis = X_cached, y_cached
        elif 'X_pool' in locals() and X_pool is not None and len(X_pool) > 0:
            X_vis, y_vis = X_pool, y_pool
        else:
            # Fallback if nothing available
            X_vis = np.zeros((1, 7 * ROW_BANDS))
            y_vis = np.zeros((1,))
            
                    # ---------------------------
        # Print summary safely (works for both cached and new runs)
        # ---------------------------
        if 'X_all' in locals() and X_all is not None:
            n_samples = len(X_all)
            n_features = X_all.shape[1]
        elif 'X_cached' in locals() and X_cached is not None:
            n_samples = len(X_cached)
            n_features = X_cached.shape[1]
        elif 'X_features' in locals() and X_for_train is not None:
            n_samples = len(X_for_train)
            n_features = X_for_train.shape[1]
        else:
            n_samples, n_features = 0, 0

        print(f"Dataset size: {n_samples} samples")
        print(f"Feature dimensionality: {n_features} -> expanded features")
        visualize_results(
            X_vis, y_vis, models, *test_data, results,
            meta=meta,
            total_seconds=total_time,
            run_tag="mnist_photonic"
        )

        # Summary
        metrics = [(k, v["test_accuracy"]) for k, v in results.items()
           if isinstance(v, dict) and "test_accuracy" in v]
        best_name, best_accuracy = max(metrics, key=lambda kv: kv[1])
        print(f"Best classifier: {best_name} (accuracy: {best_accuracy:.3f})")

        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Best classification accuracy: {best_accuracy:.1%}")
        print(f"Dataset size: {(n_samples)} samples")
        print(f"Feature dimensionality: {n_features} -> expanded features")
        
        # Save classifier for future use
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build full save path
        save_path = os.path.join("MNIST","models", f"mnist_photonic_classifier_{timestamp}.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'models': models,
                'results': results,
                'config': {
                    'INPUT_HEATERS': INPUT_HEATERS,
                    'SPATIAL_GAIN': SPATIAL_GAIN,
                }
            }, f)
        print("Classifier saved to 'mnist_photonic_classifier.pkl'")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if reservoir is not None:
            try:
                reservoir.close()
            except Exception:
                pass

if __name__ == "__main__":  
    main_mnist()
