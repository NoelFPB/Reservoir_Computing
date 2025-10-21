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

SCOPE1_CHANNELS = [1, 2, 3, 4]   # first scope (4 channels)
SCOPE2_CHANNELS = [1, 2, 3]      # second scope (3 channels)

INPUT_HEATERS = [28, 29, 30, 31, 32, 33, 34]
ALL_HEATERS = list(range(35))  # Omitting the second part of C
V_MIN, V_MAX = 0.10, 4.90
V_BIAS_INTERNAL = 2.50
V_BIAS_INPUT = 2.50

ROW_BANDS = 7 # How many 7-wide row bands to use (2, 3, or 4 recommended)
K_VIRTUAL = 1            # Still use virtual nodes for feature diversity

READ_AVG = 1             # Fewer averages needed
# Spatial encoding parameters
SPATIAL_GAIN = 0     # How strongly pixels drive heaters

# Dataset parameters
N_SAMPLES_PER_DIGIT = 1000 # Samples per digit class (500 total for quick demo)
TEST_FRACTION = 0.2      # 20% for testing
      
def zero_mean_orthogonal_masks(k, width=7, seed=42, max_tries=5000):
    rng = np.random.default_rng(seed)
    masks = []
    ones = np.ones(width)

    def is_ok(vec):
        # +/-1 entries, near-zero sum, and roughly orthogonal to existing rows and the all-ones vector
        if abs(vec.sum()) > 1:
            return False
        if abs((vec @ ones) / width) > 0.2:
            return False
        for existing in masks:
            if abs((vec @ existing) / width) > 0.2:
                return False
        return True

    for _ in range(max_tries):
        vec = rng.choice([-1.0, 1.0], size=width)
        # Keep counts of +1/-1 within one of each other
        if abs((vec == 1).sum() - (vec == -1).sum()) > 1:
            continue
        if is_ok(vec):
            masks.append(vec)
            if len(masks) == k:
                break

    if len(masks) < k:
        # Fallback: Gram-Schmidt projection with sign forcing for remaining rows
        data = np.array(masks, float)
        while len(masks) < k:
            vec = rng.standard_normal(width)
            if data.size:
                vec = vec - data.T @ np.linalg.pinv(data @ data.T) @ (data @ vec)
            vec = np.sign(vec)
            if is_ok(vec):
                masks.append(vec)
                data = np.array(masks, float)
    return np.array(masks, float)


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
    col_reduced = img2d.reshape(28, 7, 4).max(axis=2)  # (28, 7)
    # rows: 28 -> M bands
    bands = np.array_split(col_reduced, M, axis=0)     # list of (rows_i, 7)
    out = np.stack([b.max(axis=0) for b in bands], axis=0)  # (M, 7)
    return out

def load_mnist_data(n_samples_per_class=50):
    print("Loading MNIST dataset...")
    try:
        # Load MNIST from sklearn
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X, y = mnist.data.values, mnist.target.values.astype(int)
        
        # Normalize pixel values to [0, 1]
        X = X / 255.0
        
        print(f"Full MNIST dataset: {X.shape[0]} samples")
        
        # Reshape images to 7x7 (49 pixels) by downsampling
        # 7 x ROW_BANDS max-pooling
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
        
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
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
        #rng = np.random.default_rng()
        # self.mesh_bias = {
        #     h: float(np.clip(V_BIAS_INTERNAL + rng.normal(0, 2.4), V_MIN, V_MAX))
        #     for h in self.internal_heaters
        # }
        self.mesh_bias = {
            "0": 4.406996127104377,
            "1": 4.890000000000001,
            "2": 2.7939843085626257,
            "3": 3.167400524158116,
            "4": 0.40262217984061266,
            "5": 1.1762001559388024,
            "6": 1.690318961996769,
            "7": 0.11,
            "8": 2.0060493078483144,
            "9": 3.117806285176484,
            "10": 3.584691978533833,
            "11": 2.613065008900305,
            "12": 3.4139593331000353,
            "13": 4.6261600000000005,
            "14": 3.6245653079369062,
            "15": 1.4075291261676262,
            "16": 1.676667068629155,
            "17": 2.952858427214628,
            "18": 0.8895707442511243,
            "19": 2.212990444047727,
            "20": 2.576543415593057,
            "21": 4.868780215068267,
            "22": 4.692905587090785,
            "23": 0.11,
            "24": 3.7495589133138827,
            "25": 2.863356703385314,
            "26": 1.413919770333326,
            "27": 2.1694327994347393
        }        
        # Input heater baseline
        # This is the non linear inout bias determined
        # self.input_bias = {
        #     28: 1.732,
        #     29: 1.764,
        #     30: 2.223,
        #     31: 2.372,
        #     32: 1.881,
        #     33: 2.436,
        #     34: 2.852}
        
        self.input_bias = {h: 0.1 for h in range(28, 35)}

        # Mask selection
        #self.mask = zero_mean_orthogonal_masks(K_VIRTUAL - 1, len(self.input_heaters), seed=42,)
        self.mask = hadamard_like_masks(K_VIRTUAL - 1, 7, seed=42)

        # Apply initial baseline
        self.bus.send({**self.mesh_bias, **self.input_bias})
        time.sleep(0.5)

    def close(self):
        self.scope.close()
        self.bus.close()

# ==========================
# PHOTONIC RESERVOIR
# ==========================

class PhotonicReservoirMNIST(PhotonicReservoir):
    """
    Photonic reservoir adapted for spatial pattern classification.
    Inherits from the time series version but modifies for spatial processing.
    """
    
    def __init__(self, input_heaters, all_heaters):
        super().__init__(input_heaters, all_heaters)
        print("[MNIST] Photonic reservoir initialized for spatial classification")
    
    def process_spatial_pattern(self, image_pixels):
        # --- config pulled from globals (unchanged) ---
        K_req   = int(globals().get("K_VIRTUAL", 1))
        readavg = int(globals().get("READ_AVG", 1))
        vmin, vmax = float(globals().get("V_MIN", 0.1)), float(globals().get("V_MAX", 4.9))
        gain    = float(globals().get("SPATIAL_GAIN", 0.25))  # now used as *mask modulation depth*

        heaters    = self.input_heaters
        chunk_size = len(heaters)  # 7

        # Ensure [0,1] pixels and flatten
        x = np.asarray(image_pixels, float).ravel()
        x = np.clip(x, 0.0, 1.0)

        # Trim to a multiple of 7
        total_pixels = (x.size // chunk_size) * chunk_size
        x = x[:total_pixels]
        num_chunks = total_pixels // chunk_size

        # --- build mask matrix ---
        if K_req <= 1:
            mask_matrix = np.ones((1, chunk_size), dtype=float)
        else:
            need = K_req - 1
            assert len(self.mask) >= need, f"Need {need} micro masks, have {len(self.mask)}"
            mask_matrix = np.vstack([
                np.ones((1, chunk_size), dtype=float),          # baseline (no dither)
                np.asarray(self.mask[:need], dtype=float)       # ±1 masks
            ])

        send = self.bus.send
        read_many = self.scope.read_many
        features = []

        for i in range(num_chunks):
            sl = slice(i*chunk_size, (i+1)*chunk_size)

            # --- direct linear mapping: 0 → vmin, 1 → vmax ---
            base = vmin + x[sl] * (vmax - vmin)   # shape (7,)

            # iterate masks; when K_req==1, this is just the baseline
            for m in mask_matrix:
                if K_req <= 1:
                    v = base
                else:
                    # multiplicative dither around the absolute base voltage
                    # gain ~ 0.1–0.3 recommended; clip to hardware limits
                    v = base * (1.0 + gain * m)

                v = np.clip(v, vmin, vmax)
                send((heaters, v))
                pd = read_many(avg=readavg)
                features.append(pd)

        return np.asarray(features, float).ravel()

    def OG_process_spatial_pattern(self, image_pixels):
 
        K_req   = int(globals().get("K_VIRTUAL", 1))        # requested K
        readavg = int(globals().get("READ_AVG", 1))
        vmin, vmax = float(globals().get("V_MIN", 0.1)), float(globals().get("V_MAX", 4.9))
        v_bias  = float(globals().get("V_BIAS_INPUT", 2.5))
        gain    = float(globals().get("SPATIAL_GAIN", 0.4))

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
                send((heaters, v))                   
                pd = read_many(avg=readavg)
                features.append(pd)

        return np.asarray(features, float).ravel()

    def process_dataset(self, X_images, y_labels, phase_name="PROCESSING"):
        """Process entire dataset. """

        print(f"[{phase_name}] Processing {len(X_images)} images...")
        X_features = []
        processed_labels = []

        for i, (image, label) in enumerate(zip(X_images, y_labels)):
            if (i + 1) % 25 == 0:
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

def build_features_classification(Z, quadratic=True, interaction=True):
    # Make sure Z is 2D
    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
    features = []
    # Bias term - make it the right shape
    features.append(np.ones((Z.shape[0], 1)))  # Shape: (n_samples, 1)
    
    # Linear terms
    features.append(Z)  # Shape: (n_samples, n_features)
    
    if quadratic:
        features.append(Z**2)  # Quadratic terms
    
    if interaction and Z.shape[1] <= 16:  # Only for manageable dimensions
        # Cross-products between photodetectors
        for i in range(Z.shape[1]):
            for j in range(i+1, Z.shape[1]):
                interaction_term = (Z[:, i] * Z[:, j]).reshape(-1, 1)  # Make it 2D
                features.append(interaction_term)
    
    return np.hstack(features)


def train_mnist_classifier(X_features, y_labels):
    """
    Train multi-class classifier for MNIST digits with feature selection and in-pipeline scaling.
    """
    print("\nTraining MNIST classifier...")
    
    # Build expanded feature set
    X_expanded = build_features_classification(X_features, quadratic=True, interaction=True)
    print(f"Feature expansion: {X_features.shape[1]} -> {X_expanded.shape[1]} features")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_expanded, y_labels, test_size=TEST_FRACTION, 
        stratify=y_labels, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    feature_count = X_expanded.shape[1]
    base_candidates = [30, 60, 90, 120, 150, 200]
    candidate_ks = sorted({k for k in base_candidates if 0 < k < feature_count})
    if not candidate_ks:
        candidate_ks = [max(1, min(feature_count, 30))]
    default_k = "all"

    logreg_pipe = Pipeline([
        ("variance", VarianceThreshold(threshold=0.0)),
        ("kbest", SelectKBest(score_func=f_classif, k=default_k)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=20000,
            random_state=42,
            solver="lbfgs"
        ))
    ])

    logreg_grid = GridSearchCV(
        logreg_pipe,
        param_grid={
            "kbest__k": candidate_ks + ["all"],
            "clf__C": [0.03, 0.1, 0.3, 1.0, 3.0]
        },
        cv=5,
        n_jobs=-1
    )

    # --- Ridge Classifier pipeline (CV inside) ---
    ridge_pipe = Pipeline([
        ("variance", VarianceThreshold(threshold=0.0)),
        ("kbest", SelectKBest(score_func=f_classif, k=default_k)),
        ("scaler", StandardScaler()),
        ("clf", RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)))
    ])

    # --- Dictionary exactly as before ---
    classifiers = {
        "Logistic Regression": logreg_grid,
        "Ridge (CV)": ridge_pipe,
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")
        
        classifier.fit(X_train, y_train)
        estimator = classifier.best_estimator_ if isinstance(classifier, GridSearchCV) else classifier
        
        train_score = estimator.score(X_train, y_train)
        test_score = estimator.score(X_test, y_test)
        y_pred = estimator.predict(X_test)
        best_params = getattr(classifier, "best_params_", None)
        
        cv_scores = cross_val_score(estimator, X_expanded, y_labels, cv=5)
        
        results[name] = {
            'classifier': estimator,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'y_pred': y_pred,
            'cv_scores': cv_scores,
            'best_params': best_params
        }
        
        print(f"{name} Results:")
        print(f"  Training accuracy: {train_score:.3f}")
        print(f"  Test accuracy: {test_score:.3f}")
        print(f"  Cross-validation: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        if best_params:
            print(f"  Best params: {best_params}")
    
    # Select best classifier
    best_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_classifier = results[best_name]['classifier']
    
    print(f"\nBest classifier: {best_name} (accuracy: {results[best_name]['test_accuracy']:.3f})")
    
    # Detailed analysis
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, results[best_name]['y_pred']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    print("\nConfusion Matrix:")
    print(cm)
    
    return best_classifier, results, (X_test, y_test)

# ==========================
# VISUALIZATION AND ANALYSIS
# ==========================

def visualize_results(X_images, y_labels, classifier, X_test, y_test, results, *,
    save_dir=os.path.join("MNIST", "figures"),
    meta=None,            # dict like {"K_VIRTUAL": 6, "ROW_BANDS": 4, "N_SAMPLES_PER_DIGIT": 200, ...}
    total_seconds=None,   # float seconds
    run_tag=None          # optional short string to add to filename
):
    
    """ Visualize classification results, annotate with run metadata, and save to disk."""
    try:
        save_dir=os.path.join("MNIST", "figures")
        # Ensure output folder exists
        os.makedirs(save_dir, exist_ok=True)
        # Build figure
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, results['Logistic Regression']['y_pred'])
        plt.imshow(cm, interpolation='nearest')  # default colormap (no explicit colors)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Per-digit accuracy
        plt.subplot(1, 2, 2)
        digit_accuracies = []
        for digit in range(10):
            digit_mask = (y_test == digit)
            if np.sum(digit_mask) > 0:
                digit_pred = results['Logistic Regression']['y_pred'][digit_mask]
                digit_true = y_test[digit_mask]
                acc = accuracy_score(digit_true, digit_pred)
                digit_accuracies.append(acc)
            else:
                digit_accuracies.append(0.0)
        plt.bar(range(10), digit_accuracies)  # default colors
        plt.xlabel('Digit')
        plt.ylabel('Accuracy')
        plt.title('Per-Digit Accuracy')
        plt.xticks(range(10))

        # --- Compose a suptitle with metadata ---
        meta = meta or {}
        best_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        best_acc  = results[best_name]['test_accuracy']

        # pull key bits if present
        K = meta.get("K_VIRTUAL", "N/A")
        bands = meta.get("ROW_BANDS", "N/A")
        nspd = meta.get("N_SAMPLES_PER_DIGIT", "N/A")
        gain = meta.get("SPATIAL_GAIN", "N/A")
        read_avg = meta.get("READ_AVG", "N/A")

        time_str = f"{total_seconds:.2f}s" if isinstance(total_seconds, (int, float)) else "N/A"
        meta_line_1 = f"Best: {best_name}  |  Test Acc: {best_acc:.3f}  |  Time: {time_str}"
        meta_line_2 = f"K={K}, ROW_BANDS={bands}, N_SAMPLES_PER_DIGIT={nspd}, SPATIAL_GAIN={gain}, READ_AVG={read_avg}"
        plt.suptitle(meta_line_1 + "\n" + meta_line_2, y=1.05, fontsize=11)
        plt.tight_layout()

        # Save figure 
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{run_tag}" if run_tag else ""
        fname_base = f"mnist_photonic_results{tag}_{ts}"
        png_path = os.path.join(save_dir, fname_base + ".png")

        plt.savefig(png_path, dpi=200, bbox_inches='tight')
        plt.show()

        print(f"[Saved] {png_path}")

    except ImportError:
        print("Matplotlib not available - skipping visualization")

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
        classifier, results, test_data = train_mnist_classifier(X_features, y_processed)
        
        # Analysis
        analyze_feature_importance(classifier)
        total_time = time.perf_counter() - t0

        meta = {
            "K_VIRTUAL": K_VIRTUAL,
            "ROW_BANDS": ROW_BANDS,
            "N_SAMPLES_PER_DIGIT": N_SAMPLES_PER_DIGIT,
            "SPATIAL_GAIN": SPATIAL_GAIN,
            "READ_AVG": READ_AVG,
        }

        visualize_results(
            X_images, y_processed, classifier, *test_data, results,
            save_dir="figures",
            meta=meta,
            total_seconds=total_time,
            run_tag="mnist_photonic"
        )

        # Summary
        best_accuracy = max([r['test_accuracy'] for r in results.values()])
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
                'classifier': classifier,
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
