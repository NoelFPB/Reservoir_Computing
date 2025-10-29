import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Lib.scope import RigolDualScopes
from Lib.heater_bus import HeaterBus
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeClassifierCV

SCOPE1_CHANNELS = [1, 2, 3, 4]   # first scope (4 channels)
SCOPE2_CHANNELS = [1, 2, 3]      # second scope (3 channels)

INPUT_HEATERS = [28, 29, 30, 31, 32, 33, 34]
ALL_HEATERS = list(range(35))  # Omitting the second part of C
V_MIN, V_MAX = 1.10, 4.50
V_BIAS_INTERNAL = 0.1
V_BIAS_INPUT = 3.4

ROW_BANDS = 4 # How many 7-wide row bands to use (2, 3, or 4 recommended)
K_VIRTUAL = 4           # Still use virtual nodes for feature diversity

READ_AVG = 1             # Fewer averages needed
# Spatial encoding parameters
SPATIAL_GAIN = 0.6     # How strongly pixels drive heaters Now should be less than 0.64

# Dataset parameters
N_SAMPLES_PER_DIGIT = 80 # Samples per digit class (500 total for quick demo)
TEST_FRACTION = 0.2      # 20% for testing
      
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
    """
    Reduce 28x28 image to a 7xM grid using max-pooling:
      - Columns: 28 -> 7 via 4-wide non-overlapping max
      - Rows:    28 -> M via max over nearly-equal row bands (np.array_split)
    Returns array shape (M, 7) where each row is one 7-value chunk.
    """
    assert img2d.shape == (28, 28)
    # columns: 28 -> 7 (keep horizontal detail for each heater)
    col_reduced = img2d.reshape(28, 7, 4).mean(axis=2)  # (28, 7)
    # rows: 28 -> M bands
    bands = np.array_split(col_reduced, M, axis=0)     # list of (rows_i, 7)
    out = np.stack([b.mean(axis=0) for b in bands], axis=0)  # (M, 7)
    return out

def load_mnist_data(n_samples_per_class=50):
    print("Loading MNIST dataset...")
    # Load MNIST from sklearn
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.values, mnist.target.values.astype(int)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    print(f"Full MNIST dataset: {X.shape[0]} samples")
    
    # Reshape images to 7x7 (49 pixels) by downsampling
    X_resized = []
    for img in X:
        img_2d = img.reshape(28, 28)
        grid7xM = downsample_to_7xM(img_2d, ROW_BANDS)  # (M,7)
        X_resized.append(grid7xM.flatten())             # size = 7*M
    X_resized = np.array(X_resized)

    # Sample balanced subset from the resized dataset
    X_subset, y_subset = create_balanced_subset(X_resized, y, n_samples_per_class)
    
    print(f"Using resized subset: {len(X_subset)} samples ({n_samples_per_class} per digit)")
    return X_subset, y_subset
    
        
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
        self.bus = HeaterBus()

        # Fixed random mesh bias

        rng = np.random.default_rng(42)
        self.mesh_bias = {
            h: float(rng.uniform(V_MIN, V_MAX))
            for h in self.internal_heaters
        }
        # Zero mesh
        # self.mesh_bias = {
        #     h: 0.1
        #     for h in self.internal_heaters
        # }

        self.input_bias = {h: V_BIAS_INPUT for h in range(28, 35)}

        self.mask = hadamard_like_masks(K_VIRTUAL - 1, 7, seed=42)

        # Apply initial baseline
        self.bus.send({**self.mesh_bias, **self.input_bias})
        print(self.input_bias)
        print(self.mesh_bias)
        #time.sleep(0.5)

    def close(self):
        self.scope.close()
        self.bus.close()

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

        send = self.bus.send
        read_many = self.scope.read_many

        features = []

        for i in range(num_chunks):
            sl = slice(i*chunk_size, (i+1)*chunk_size)
            base = gain * half_swing * x_centered[sl]            # shape (7,)
            # iterate masks (K_eff times; when K_eff==1 this is minimal)
            for m in mask_matrix:
                v = v_bias + base * m
                v = np.clip(v, vmin, vmax)
                print("v", np.round(v, 2))
                send((heaters, v))                   
                pd = read_many(avg=readavg)
                features.append(pd)

        return np.asarray(features, float).ravel()

    def process_dataset(self, X_images, y_labels, phase_name="PROCESSING"):

        print(f"[{phase_name}] Processing {len(X_images)} images...")
        X_features = []
        processed_labels = []

        for i, (image, label) in enumerate(zip(X_images, y_labels)):
            if (i + 1) % 10 == 0:
                print(f"[{phase_name}] Processed {i+1}/{len(X_images)} images")

            try:
                features = self.process_spatial_pattern(image)
                # Check for NaNs and skip if found
                if not np.any(np.isnan(features)):
                    X_features.append(features)
                    processed_labels.append(label)
                else:
                    print(f"[WARNING] Skipping image {i} due to NaN values in features.")

            except Exception as e:
                print(f"[ERROR] Failed to process image {i}: {e}")
                continue

        X_features = np.array(X_features)
        processed_labels = np.array(processed_labels)

        print(f"[{phase_name}] Completed: {X_features.shape[0]} samples, {X_features.shape[1]} features")

        return X_features, processed_labels

# ==========================
# CLASSIFICATION TRAINING
# ==========================

def train_mnist_classifier(X_features, y_labels, *, seed=42):
    """
    Trains TWO one-layer heads on fixed features (ELM style):
      1) LinearRegression (multi-output on one-hot; predict via argmax)
      2) RidgeClassifierCV (direct linear classifier on labels)

    Returns:
      models = {
        "linreg": linear_regression_model,
        "ridge_clf": ridge_classifier_model,
      },
      results = {
        "linreg": {...},
        "ridge_clf": {...},
        "scaler": StandardScaler fitted on train,
      },
      (X_test, y_test)
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
                      save_dir=os.path.join("MNIST", "figures"),
                      meta=None, total_seconds=None, run_tag=None):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, accuracy_score

    # --- choose which model's predictions to plot ---
    # Prefer ridge_clf if available; otherwise first entry that has y_pred
    if "ridge_clf" in results and isinstance(results["ridge_clf"], dict) and "y_pred" in results["ridge_clf"]:
        key = "ridge_clf"
    else:
        key = next(k for k, v in results.items() if isinstance(v, dict) and "y_pred" in v)

    y_pred = np.asarray(results[key]["y_pred"])

    # ensure output folder exists
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
        # Load data
        X_images, y_labels = load_mnist_data(N_SAMPLES_PER_DIGIT)
        print(f"Dataset loaded: {len(X_images)} samples, {len(np.unique(y_labels))} classes")

        reservoir = PhotonicReservoirMNIST(INPUT_HEATERS, ALL_HEATERS)

        # Process dataset through reservoir
        X_features, y_processed = reservoir.process_dataset(X_images, y_labels, "TRAINING")

        # Free hardware resources as soon as possible
        if reservoir is not None:
            reservoir.close()
            reservoir = None
        
        if len(X_features) == 0:
            print("ERROR: No samples processed successfully!")
            return
        
        # Train classifier
        models, results, test_data = train_mnist_classifier(X_features, y_processed)
        
        # add this right after the line above
        scaler = results.pop("scaler", None)

        # Analysis
        clf_for_importance = models.get("ridge_clf", models.get("linreg"))
        if clf_for_importance is not None:
            analyze_feature_importance(clf_for_importance)

        total_time = time.perf_counter() - t0

        meta = {
            "K_VIRTUAL": K_VIRTUAL,
            "ROW_BANDS": ROW_BANDS,
            "N_SAMPLES_PER_DIGIT": N_SAMPLES_PER_DIGIT,
            "SPATIAL_GAIN": SPATIAL_GAIN,
            "READ_AVG": READ_AVG,
        }

        visualize_results(
            X_images, y_processed, models, *test_data, results,
            save_dir="figures",
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
        print(f"Dataset size: {len(X_images)} samples")
        print(f"Feature dimensionality: {X_features.shape[1]} -> expanded features")
        
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
                    'K_VIRTUAL': K_VIRTUAL
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
