import time, serial, pyvisa
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from scope import  RigolScope

"""
MNIST Digit Classification using Photonic Reservoir Computing

Takes the same hardware used for Mackey-Glass prediction and adapts it 
for handwritten digit recognition (0-9 classification).

Key insight: Same physical reservoir, different interpretation!
"""

# ==========================
# HELPER FUNCTIONS (from original reservoir code)
# ==========================

def rand_mask(n):
    """Balanced ±1 mask of length n."""
    m = np.ones(n)
    m[np.random.choice(n, size=n//2, replace=False)] = -1
    return m

def build_features(Z, quadratic=True):
    """Features = [1, Z, Z^2] (or [1, Z])."""
    if quadratic:
        return np.hstack([np.ones((len(Z),1)), Z, Z**2])
    return np.hstack([np.ones((len(Z),1)), Z])

# ==========================
# CONFIG (Same as before)
# ==========================
SERIAL_PORT = 'COM3'
BAUD_RATE = 112500

SCOPE_CHANNELS = [1, 2, 3, 4]
INPUT_HEATERS = [33, 34, 35, 36, 37, 38, 39]
ALL_HEATERS = list(range(40))

V_MIN, V_MAX = 0.10, 4.90
V_BIAS_INTERNAL = 2.50
V_BIAS_INPUT = 2.50

# Modified timing for spatial patterns (can be faster since no temporal sequence)
T_SETTLE = 0.2          # Time to let spatial pattern develop
K_VIRTUAL = 4            # Still use virtual nodes for feature diversity
SETTLE = 0.05            # Faster sampling for spatial patterns, how ofter we measure the nodes
READ_AVG = 1             # Fewer averages needed

# Spatial encoding parameters
SPATIAL_GAIN = 5.0       # How strongly pixels drive heaters
NOISE_LEVEL = 0.05        # Add slight randomization to prevent overfitting

# Dataset parameters
N_SAMPLES_PER_DIGIT = 40 # Samples per digit class (500 total for quick demo)
TEST_FRACTION = 0.2      # 20% for testing

# ==========================
# DATA LOADING AND PREPROCESSING
# ==========================

def load_mnist_data(n_samples_per_class=50):
    """Load and preprocess MNIST data."""
    print("Loading MNIST dataset...")
    
    try:
        # Load MNIST from sklearn
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X, y = mnist.data.values, mnist.target.values.astype(int)
        
        # Normalize pixel values to [0, 1]
        X = X / 255.0
        
        print(f"Full MNIST dataset: {X.shape[0]} samples")
        
        # Sample balanced subset
        X_subset, y_subset = create_balanced_subset(X, y, n_samples_per_class)
        
        print(f"Using subset: {len(X_subset)} samples ({n_samples_per_class} per digit)")
        return X_subset, y_subset
        
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Generating synthetic digit-like data for demo...")
        return generate_synthetic_digits(n_samples_per_class)

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

def generate_synthetic_digits(n_per_class):
    """Generate synthetic digit-like patterns for testing."""
    print("Generating synthetic digit data...")
    
    X_synthetic = []
    y_synthetic = []
    
    for digit in range(10):
        for _ in range(n_per_class):
            # Create simple digit-like patterns
            pattern = np.zeros(784)  # 28x28 flattened
            
            if digit == 0:  # Circle-like
                pattern[200:250] = np.random.uniform(0.5, 1.0, 50)
                pattern[500:550] = np.random.uniform(0.5, 1.0, 50)
            elif digit == 1:  # Vertical line
                pattern[100:200:10] = np.random.uniform(0.7, 1.0, 10)
            # ... (can add more patterns)
            else:  # Random pattern for other digits
                random_indices = np.random.choice(784, size=50, replace=False)
                pattern[random_indices] = np.random.uniform(0.3, 1.0, 50)
            
            # Add noise
            pattern += np.random.normal(0, 0.1, 784)
            pattern = np.clip(pattern, 0, 1)
            
            X_synthetic.append(pattern)
            y_synthetic.append(digit)
    
    return np.array(X_synthetic), np.array(y_synthetic)

# ==========================
# HARDWARE CLASSES (from original reservoir code)
# ==========================

class HeaterBus:
    """Serial sender for 'heater,value;...\\n' strings."""
    def __init__(self):
        
        print(f"[DEBUG] Connecting to serial port {SERIAL_PORT}...")
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send(self, config: dict):
        msg = "".join(f"{h},{float(v):.3f};" for h,v in config.items()) + "\n"
        
        self.ser.write(msg.encode())
        self.ser.flush()

    def close(self):
        try: 
            self.ser.close()
        except: 
            pass

class PhotonicReservoir:
    """Base photonic reservoir class."""
    def __init__(self, input_heaters, all_heaters, scope_channels):
        self.input_heaters = list(input_heaters)
        self.internal_heaters = [h for h in all_heaters if h not in self.input_heaters]
        self.scope = RigolScope(scope_channels)
        self.bus = HeaterBus()

        # Fixed random mesh bias
        rng = np.random.default_rng()
        self.mesh_bias = {
            h: float(np.clip(V_BIAS_INTERNAL + rng.normal(0, 1.5), V_MIN, V_MAX))
            for h in self.internal_heaters
        }

        print(self.mesh_bias)
        
        # Input heater baseline
        self.input_bias = {h: V_BIAS_INPUT for h in self.input_heaters}

        # Input masks (fixed)
        self.main_mask = rand_mask(len(self.input_heaters))
        self.micro_masks = [rand_mask(len(self.input_heaters)) for _ in range(K_VIRTUAL)]

        # Apply initial baseline
        self.bus.send({**self.mesh_bias, **self.input_bias})
        time.sleep(0.2)

    def close(self):
        self.scope.close()
        self.bus.close()

# ==========================
# SPATIAL ENCODING
# ==========================

def encode_image_to_heaters(image_pixels):
    """
    Convert 784 pixel values to 7 heater voltages.
    Uses spatial pooling and random projections.
    """
    # Method 1: Spatial pooling - divide image into regions
    heater_values = []
    pixels_per_heater = len(image_pixels) // len(INPUT_HEATERS)
    
    for i in range(len(INPUT_HEATERS)):
        start_idx = i * pixels_per_heater
        end_idx = start_idx + pixels_per_heater
        
        # Average pixel intensity in this region
        region_intensity = np.mean(image_pixels[start_idx:end_idx])
        
        # Convert to heater voltage with noise for regularization
        base_voltage = V_BIAS_INPUT + SPATIAL_GAIN * (region_intensity - 0.5)
        noise = np.random.normal(0, NOISE_LEVEL)
        heater_voltage = base_voltage + noise
        
        # Clip to safe range
        heater_values.append(np.clip(heater_voltage, V_MIN, V_MAX))
    
    return heater_values

def encode_image_advanced(image_pixels):
    """
    Advanced encoding using random projections for better feature extraction.
    """
    # Reshape to 28x28 for spatial operations
    image_2d = image_pixels.reshape(28, 28)
    
    heater_values = []
    
    for i, heater in enumerate(INPUT_HEATERS):
        # Create different spatial filters for each heater
        if i == 0:  # Top edge detector
            region = image_2d[:10, :].flatten()
        elif i == 1:  # Bottom edge detector  
            region = image_2d[-10:, :].flatten()
        elif i == 2:  # Left edge detector
            region = image_2d[:, :10].flatten()
        elif i == 3:  # Right edge detector
            region = image_2d[:, -10:].flatten()
        elif i == 4:  # Center region
            region = image_2d[9:19, 9:19].flatten()
        elif i == 5:  # Diagonal 1
            region = np.diag(image_2d).flatten()
        else:  # Diagonal 2 or random projection
            region = np.diag(np.fliplr(image_2d)).flatten()
        
        # Compute intensity for this spatial filter
        region_intensity = np.mean(region)
        
        # Convert to voltage
        heater_voltage = V_BIAS_INPUT + SPATIAL_GAIN * (region_intensity - 0.5)
        heater_voltage += np.random.normal(0, NOISE_LEVEL)  # Regularization
        
        heater_values.append(np.clip(heater_voltage, V_MIN, V_MAX))
    
    return heater_values

# ==========================
# PHOTONIC RESERVOIR (Modified for Spatial Processing)
# ==========================

class PhotonicReservoirMNIST(PhotonicReservoir):
    """
    Photonic reservoir adapted for spatial pattern classification.
    Inherits from the time series version but modifies for spatial processing.
    """
    
    def __init__(self, input_heaters, all_heaters, scope_channels):
        super().__init__(input_heaters, all_heaters, scope_channels)
        print("[MNIST] Photonic reservoir initialized for spatial classification")
    
    def process_spatial_pattern(self, image_pixels, encoding_method='advanced'):
        """
        Process a single spatial pattern (image) through the reservoir.
        Returns feature vector for classification.
        """
        # Encode image to heater pattern
        if encoding_method == 'advanced':
            heater_voltages = encode_image_advanced(image_pixels)
        else:
            heater_voltages = encode_image_to_heaters(image_pixels)
        
        # Create full configuration
        config = dict(self.mesh_bias)  # Internal mesh stays random
        for i, heater in enumerate(self.input_heaters):
            config[heater] = heater_voltages[i]
        
        # Apply spatial pattern and let it develop
        self.bus.send(config)
        time.sleep(T_SETTLE)  # Let thermal pattern develop
        
        # Collect features using virtual nodes (temporal multiplexing)
        features = []
        for k in range(K_VIRTUAL):
            time.sleep(SETTLE)  # Small delay between samples
            pd_reading = self.scope.read_many(avg=READ_AVG)
            features.extend(pd_reading)
        
        return np.array(features)
    
    def process_dataset(self, X_images, y_labels, phase_name="PROCESSING"):
        """
        Process entire dataset through the reservoir.
        """
        print(f"[{phase_name}] Processing {len(X_images)} images...")
        
        X_features = []
        processed_labels = []
        
        for i, (image, label) in enumerate(zip(X_images, y_labels)):
            if (i + 1) % 50 == 0:
                print(f"[{phase_name}] Processed {i+1}/{len(X_images)} images")
            
            try:
                features = self.process_spatial_pattern(image)
                X_features.append(features)
                processed_labels.append(label)
                
                # Debug info for first few samples
                if i < 3:
                    print(f"[DEBUG] Image {i} (digit {label}): features shape {features.shape}")
                    print(f"         Feature range: {features.min():.3f} to {features.max():.3f}")
                
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
    """
    Enhanced feature building for classification.
    """
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
    Train multi-class classifier for MNIST digits.
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
    
    # Try different classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Ridge Classifier': Ridge(alpha=1.0)
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train
        classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        
        # Predictions for detailed analysis
        y_pred = classifier.predict(X_test)
        
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
        cv_scores = cross_val_score(classifier, X_expanded, y_labels, cv=5)
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

def visualize_results(X_images, y_labels, classifier, X_test, y_test, results):
    """
    Visualize classification results.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 4))
        
        # Confusion matrix
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_test, results['Logistic Regression']['y_pred'])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Sample classifications
        plt.subplot(1, 3, 2)
        # Find some example images and their classifications
        sample_indices = np.random.choice(len(X_test), 6, replace=False)
        # This would require mapping back to original images
        plt.title('Sample Classifications')
        plt.text(0.5, 0.5, 'See console output\nfor detailed results', 
                ha='center', va='center', transform=plt.gca().transAxes)
        
        # Accuracy by digit
        plt.subplot(1, 3, 3)
        digit_accuracies = []
        for digit in range(10):
            digit_mask = (y_test == digit)
            if np.sum(digit_mask) > 0:
                digit_pred = results['Logistic Regression']['y_pred'][digit_mask]
                digit_true = y_test[digit_mask]
                accuracy = accuracy_score(digit_true, digit_pred)
                digit_accuracies.append(accuracy)
            else:
                digit_accuracies.append(0)
        
        plt.bar(range(10), digit_accuracies)
        plt.xlabel('Digit')
        plt.ylabel('Accuracy')
        plt.title('Per-Digit Accuracy')
        plt.xticks(range(10))
        
        plt.tight_layout()
        plt.show()
        
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
    """
    Main function for MNIST classification with photonic reservoir.
    """
    print("="*60)
    print("PHOTONIC RESERVOIR MNIST CLASSIFICATION")
    print("="*60)
    
    try:
        # Load data
        X_images, y_labels = load_mnist_data(N_SAMPLES_PER_DIGIT)
        print(f"Dataset loaded: {len(X_images)} samples, {len(np.unique(y_labels))} classes")
        
        # Initialize reservoir
        reservoir = PhotonicReservoirMNIST(INPUT_HEATERS, ALL_HEATERS, SCOPE_CHANNELS)
        
        # Process dataset through reservoir
        X_features, y_processed = reservoir.process_dataset(X_images, y_labels, "TRAINING")
        
        if len(X_features) == 0:
            print("ERROR: No samples processed successfully!")
            return
        
        # Train classifier
        classifier, results, test_data = train_mnist_classifier(X_features, y_processed)
        
        # Analysis
        analyze_feature_importance(classifier)
        
        # Visualization
        visualize_results(X_images, y_processed, classifier, *test_data, results)
        
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

def demo_single_digit():
    """
    Quick demo classifying a single digit.
    """
    print("Quick single-digit classification demo...")
    
    # Generate or load a single digit
    X_demo, y_demo = load_mnist_data(1)  # Just one sample
    
    if len(X_demo) > 0:
        reservoir = PhotonicReservoirMNIST(INPUT_HEATERS, ALL_HEATERS, SCOPE_CHANNELS)
        
        print(f"Processing digit {y_demo[0]}...")
        features = reservoir.process_spatial_pattern(X_demo[0])
        
        print(f"Reservoir response: {len(features)} features")
        print(f"Feature range: {features.min():.3f} to {features.max():.3f}")
        print(f"Feature std: {features.std():.3f}")
        
        reservoir.close()

if __name__ == "__main__":
    # For quick testing, run single digit demo
    # demo_single_digit()
    
    # For full MNIST classification
    main_mnist()