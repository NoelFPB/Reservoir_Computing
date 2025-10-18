import time, serial, pyvisa
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from  Lib.scope import  RigolDualScopes
from Lib.heater_bus import HeaterBus
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
import os, time, json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# MAYBE TRY OPTION A FOR THE MASK THING

"""
MNIST Digit Classification using Photonic EML

"""

# ==========================
# CONFIG
# ==========================
# How many 7-wide row bands to use (2, 3, or 4 recommended)
ROW_BANDS = 4   # 3 → 21 pixels → 3 chunks per image
# When using chunk mode, keep projection off:
PROJECTION_MODE = False

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

SCOPE1_CHANNELS = [1, 2, 3, 4]   # first scope (4 channels)
SCOPE2_CHANNELS = [1, 2, 3]      # second scope (3 channels)


INPUT_HEATERS = [28, 29, 30, 31, 32, 33, 34]
ALL_HEATERS = list(range(35)) # Ommitting the second part of C
V_MIN, V_MAX = 0.10, 4.90
V_BIAS_INTERNAL = 2.50
V_BIAS_INPUT = 2.50

# Modified timing for spatial patterns (can be faster since no temporal sequence)
T_SETTLE = 0.03          # Time to let spatial pattern develop
K_VIRTUAL = 3            # Still use virtual nodes for feature diversity
SETTLE = 0.006            # Faster sampling for spatial patterns, how ofter we measure the nodes
READ_AVG = 1             # Fewer averages needed
BURST = 1
# Spatial encoding parameters
SPATIAL_GAIN = 0.5       # How strongly pixels drive heaters
NOISE_LEVEL = 0.05        # Add slight randomization to prevent overfitting

# Dataset parameters
N_SAMPLES_PER_DIGIT = 250 # Samples per digit class (500 total for quick demo)
TEST_FRACTION = 0.2      # 20% for testing
      

def zero_mean_orthogonal_masks(k, width=7, seed=42, max_tries=5000):
    rng = np.random.default_rng(seed)
    M = []
    ones = np.ones(width)

    def is_ok(v):
        # ±1, near-zero-sum and orthogonal-enough to existing rows & ones
        if abs(v.sum()) > 1:               # keep |sum| ≤ 1
            return False
        if abs((v @ ones) / width) > 0.2:  # ~orthogonal to ones
            return False
        for m in M:
            if abs((v @ m) / width) > 0.2: # ~orthogonal to previous masks
                return False
        return True

    for _ in range(max_tries):
        v = rng.choice([-1.0, 1.0], size=width)
        # Heuristic balance: force counts of +1/-1 to differ by ≤1
        if abs((v == 1).sum() - (v == -1).sum()) > 1:
            continue
        if is_ok(v):
            M.append(v)
            if len(M) == k:
                break

    if len(M) < k:
        # fallback: Gram-Schmidt + sign for last few
        X = np.array(M, float)
        # Project random vectors to orthogonal subspace, then sign
        while len(M) < k:
            v = rng.standard_normal(width)
            if X.size:
                v = v - X.T @ np.linalg.pinv(X @ X.T) @ (X @ v)
            v = np.sign(v)
            if is_ok(v):
                M.append(v)
                X = np.array(M, float)
    return np.array(M, float)


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
# FUNCTIONS
# ==========================

def rand_mask(n):
    """Balanced ±1 mask of length n."""
    m = np.ones(n)
    m[np.random.choice(n, size=n//2, replace=False)] = -1
    return m

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
        rng = np.random.default_rng()
        # self.mesh_bias = {
        #     h: float(np.clip(V_BIAS_INTERNAL + rng.normal(0, 2.4), V_MIN, V_MAX))
        #     for h in self.internal_heaters
        # }
        self.mesh_bias = {
            "0": 4.796350930812443,
            "1": 1.6076502536270239,
            "2": 2.383511208014088,
            "3": 3.551003935907767,
            "4": 4.053483654047246,
            "5": 0.670394543914777,
            "6": 3.715578857040116,
            "7": 1.5323928915358995,
            "8": 3.7097140270922657,
            "9": 4.890000000000001,
            "10": 1.3230155149724867,
            "11": 4.0041804961291065,
            "12": 3.1757363250853508,
            "13": 2.58429098448964,
            "14": 2.219189520661344,
            "15": 4.424411576783365,
            "16": 4.890000000000001,
            "17": 1.5651924903701682,
            "18": 4.49,
            "19": 1.187314346890345,
            "20": 2.377810818609803,
            "21": 2.8891189414236087,
            "22": 3.7498969658506893,
            "23": 4.81713621727373,
            "24": 4.588722210119075,
            "25": 2.586307284997763,
            "26": 4.181690348756786,
            "27": 2.212052055752035
        }
        #self.mesh_bias = {h: 0.0 for h in self.internal_heaters}
        print(self.mesh_bias)
        
        # Input heater baseline
        #self.input_bias = {h: V_BIAS_INPUT for h in self.input_heaters}

        #This is the non linear inout bias determined
        self.input_bias = {
            28: 1.732,
            29: 1.764,
            30: 2.223,
            31: 2.372,
            32: 1.881,
            33: 2.436,
            34: 2.852}

        # Input masks (fixed)
        self.main_mask = rand_mask(len(self.input_heaters))
        #self.micro_masks = [rand_mask(len(self.input_heaters)) for _ in range(K_VIRTUAL)]
        # in __init__:
        #self.micro_masks = hadamard_like_masks(K_VIRTUAL-1, len(self.input_heaters), seed=42)

        # EITHER: no ones-row, K masks total
        #self.micro_masks = zero_mean_orthogonal_masks(K_VIRTUAL, len(self.input_heaters), seed=42)
        # and remove the special-casing that inserts the all-ones row

        # OR: keep the ones-row, but generate K-1 micro-masks orthogonal to ones:
        self.micro_masks = zero_mean_orthogonal_masks(K_VIRTUAL - 1, len(self.input_heaters), seed=42)



        # Apply initial baseline
        self.bus.send({**self.mesh_bias, **self.input_bias})
        time.sleep(0.2)

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
        """
        Unified logic for all K:
        - builds a mask matrix of shape (K_eff, 7)
            * K_eff = 1  -> [ones]           (fast path, same behavior as old code)
            * K_eff > 1 -> [ones] + (K_eff-1) random ±1 masks from self.micro_masks
        - one write + one read per mask per chunk
        - no autoscale, no retry, BURST=1 (keep READ_AVG small for speed)

        Returns a 1D feature vector of length: num_chunks * K_eff * n_channels
        """
        # ---- globals / params ----
        K_req   = int(globals().get("K_VIRTUAL", 1))        # requested K
        settle  = float(globals().get("SETTLE", 0.0))
        readavg = int(globals().get("READ_AVG", 1))
        vmin, vmax = float(globals().get("V_MIN", 0.1)), float(globals().get("V_MAX", 4.9))
        v_bias  = float(globals().get("V_BIAS_INPUT", 2.5))
        gain    = float(globals().get("SPATIAL_GAIN", 0.4))

        heaters     = self.input_heaters
        chunk_size  = len(heaters)                # 7
        n_channels  = len(self.scope.channels)

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
            assert len(self.micro_masks) >= need, \
                f"Need {need} micro masks, have {len(self.micro_masks)}"
            mask_matrix = np.vstack([
                np.ones((1, chunk_size), dtype=float),             # baseline mask
                np.asarray(self.micro_masks[:need], dtype=float)   # ±1 masks
            ])
        K_eff = mask_matrix.shape[0]

        # ---- local refs reduce Python overhead ----
        send      = self.bus.send
        read_many = self.scope.read_many

        # tiny spin-wait for sub-ms delays; otherwise sleep
        def tiny_wait(dt):
            if dt <= 0.0: return
            if dt < 0.001:
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < dt:
                    pass
            else:
                time.sleep(dt)

        features = []

        for i in range(num_chunks):
            # per-chunk base voltage offset (no autoscale for speed)
            sl = slice(i*chunk_size, (i+1)*chunk_size)
            base = gain * half_swing * x_centered[sl]            # shape (7,)

            # iterate masks (K_eff times; when K_eff==1 this is minimal)
            for m in mask_matrix:
                v = v_bias + base * m
                v = np.clip(v, vmin, vmax)

                # try a faster bus path (tuple/array); else fallback to dict
                try:
                    send((heaters, v))                           # if your driver supports it
                except TypeError:
                    send({h: float(f"{vj:.3f}") for h, vj in zip(heaters, v)})

                tiny_wait(settle)
                pd = read_many(avg=readavg)
                features.append(pd)

        return np.asarray(features, float).ravel()

    def process_dataset(self, X_images, y_labels, phase_name="PROCESSING"):
        """
        Process entire dataset through the reservoir.
        """
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
    Train multi-class classifier for MNIST digits with data scaling.
    """
    print("\nTraining MNIST classifier...")
    
    # Build expanded feature set
    X_expanded = build_features_classification(X_features, quadratic=True, interaction=False)
    print(f"Feature expansion: {X_features.shape[1]} → {X_expanded.shape[1]} features")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_expanded, y_labels, test_size=TEST_FRACTION, 
        stratify=y_labels, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # --- Data Scaling Implementation ---
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Use the same scaler to transform the test data
    X_test_scaled = scaler.transform(X_test)

    # Note: For cross-validation, the scaler must be applied inside the CV loop
    # or a pipeline should be used to prevent data leakage.
    
    # Try different classifiers
    # classifiers = {
    #     'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    #     'Ridge Classifier': Ridge(alpha=1.0)
    # }

    # classifiers = {
    #     'Logistic Regression': LogisticRegression(
    #         max_iter=10000,
    #         random_state=42,
    #         solver='lbfgs',
    #         C=0.1              # good for multinomial
    #     ),
    #         'Ridge (CV)': 
    #     RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))}
    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=20000,
            random_state=42,
            solver="lbfgs",
        ))
    ])

    logreg_grid = GridSearchCV(
        logreg_pipe,
        param_grid={"clf__C": [0.03, 0.1, 0.3, 1.0]},
        cv=5,
        n_jobs=-1
    )

    # --- Ridge Classifier pipeline (CV inside) ---
    ridge_pipe = Pipeline([
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
            
            # Train on scaled data
            classifier.fit(X_train_scaled, y_train)
            
            # Evaluate on scaled data
            train_score = classifier.score(X_train_scaled, y_train)
            test_score = classifier.score(X_test_scaled, y_test)
            
            # Predictions for detailed analysis
            y_pred = classifier.predict(X_test_scaled)
            
            # # Fix for Ridge Classifier: Convert continuous output to integer labels
            # if name == 'Ridge Classifier':
            #     y_pred = np.rint(y_pred).astype(int) # rounds and casts to int
                
            results[name] = {
                'classifier': classifier,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'y_pred': y_pred
            }
            
            print(f"{name} Results:")
            print(f"  Training accuracy: {train_score:.3f}")
            print(f"  Test accuracy: {test_score:.3f}")
            
            # Cross-validation
            # Using a pipeline to correctly scale data within each fold
  
            pipeline = make_pipeline(StandardScaler(), classifier)
            cv_scores = cross_val_score(pipeline, X_expanded, y_labels, cv=5)
            print(f"  Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
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



def visualize_results(
    X_images, y_labels, classifier, X_test, y_test, results,
    *,
    save_dir="figures",
    meta=None,            # dict like {"K_VIRTUAL": 6, "ROW_BANDS": 4, "N_SAMPLES_PER_DIGIT": 200, ...}
    total_seconds=None,   # float seconds
    run_tag=None          # optional short string to add to filename
):
    """
    Visualize classification results, annotate with run metadata, and save to disk.
    """
    try:
        import matplotlib.pyplot as plt

        # Ensure output folder exists
        os.makedirs(save_dir, exist_ok=True)

        # Build figure
        plt.figure(figsize=(12, 4))

        # Confusion matrix (use the best model in results if you want; keeping logistic here to match your code)
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

        # --- Save figure (timestamped filename) ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{run_tag}" if run_tag else ""
        fname_base = f"mnist_photonic_results{tag}_{ts}"
        png_path = os.path.join(save_dir, fname_base + ".png")

        plt.savefig(png_path, dpi=200, bbox_inches='tight')

        # (Optional) save a small metadata JSON next to the image
        meta_out = dict(meta)
        meta_out.update({
            "best_classifier": best_name,
            "test_accuracy": float(best_acc),
            "total_seconds": float(total_seconds) if isinstance(total_seconds, (int, float)) else None,
            "timestamp": ts,
            "filename": os.path.basename(png_path),
        })
        with open(os.path.join(save_dir, fname_base + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)

        # Show (after saving to ensure it’s written even if the window is closed)
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

    print(ALL_HEATERS)
    """
    Main function for MNIST classification with photonic reservoir.
    """
    print("="*60)
    print("PHOTONIC RESERVOIR MNIST CLASSIFICATION")
    print("="*60)
    t0 = time.perf_counter()
    try:
        # Load data
        X_images, y_labels = load_mnist_data(N_SAMPLES_PER_DIGIT)
        print(f"Dataset loaded: {len(X_images)} samples, {len(np.unique(y_labels))} classes")
        
        # Initialize reservoir
        reservoir = PhotonicReservoirMNIST(INPUT_HEATERS, ALL_HEATERS)
        
        # Process dataset through reservoir
        X_features, y_processed = reservoir.process_dataset(X_images, y_labels, "TRAINING")
        
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
            "SETTLE": SETTLE,
            "T_SETTLE": T_SETTLE,
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
        print(f"Feature dimensionality: {X_features.shape[1]} → expanded features")
        print(f"Processing time per sample: ~{(T_SETTLE + K_VIRTUAL*SETTLE):.2f}s")
        
        # Save classifier for future use
        import pickle
        with open('mnist_photonic_classifier.pkl', 'wb') as f:
            pickle.dump({
                'classifier': classifier,
                'results': results,
                'config': {
                    'INPUT_HEATERS': INPUT_HEATERS,
                    'SPATIAL_GAIN': SPATIAL_GAIN,
                    'T_SETTLE': T_SETTLE,
                    'K_VIRTUAL': K_VIRTUAL
                }
            }, f)
        print("Classifier saved to 'mnist_photonic_classifier.pkl'")
        
        reservoir.close()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            reservoir.close()
        except:
            pass

        
        reservoir.close()

if __name__ == "__main__":  
    # For full MNIST classification
    main_mnist()