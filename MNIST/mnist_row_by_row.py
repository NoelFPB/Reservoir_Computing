import time, serial, pyvisa
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from scope import  RigolScope
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

"""
MNIST Digit Classification using Photonic Reservoir Computing

"""

# ==========================
# CONFIG
# ==========================
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

SCOPE_CHANNELS = [1, 2, 3, 4]
INPUT_HEATERS = [33, 34, 35, 36, 37, 38, 39]
ALL_HEATERS = list(range(40))

V_MIN, V_MAX = 0.10, 4.90
V_BIAS_INTERNAL = 2.50
V_BIAS_INPUT = 2.50

# Modified timing for spatial patterns (can be faster since no temporal sequence)
T_SETTLE = 0.1          # Time to let spatial pattern develop
K_VIRTUAL = 2            # Still use virtual nodes for feature diversity
SETTLE = 0.02            # Faster sampling for spatial patterns, how ofter we measure the nodes
READ_AVG = 1             # Fewer averages needed

# Spatial encoding parameters
SPATIAL_GAIN = 5.0       # How strongly pixels drive heaters
NOISE_LEVEL = 0.05        # Add slight randomization to prevent overfitting

# Dataset parameters
N_SAMPLES_PER_DIGIT = 20 # Samples per digit class (500 total for quick demo)
TEST_FRACTION = 0.2      # 20% for testing
      
# ==========================
# DATA LOADING AND PREPROCESSING
# ==========================


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
        X_resized = []
        for img in X:
            # Reshape to a 28x28 grid
            img_2d = img.reshape(28, 28)
            # Reshape for downsampling to 7x7 (4x4 blocks)
            img_downsampled = img_2d.reshape(7, 4, 7, 4).mean(axis=(1, 3))
            # Flatten back to a 1D array and append
            X_resized.append(img_downsampled.flatten())

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

class HeaterBus:
    """Serial sender for 'heater,value;...\\n' strings."""
    def __init__(self):
        
        print(f"Connecting to serial port {SERIAL_PORT}...")
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send(self, config):
          # Create a new dictionary with standard Python floats for printing
        # printable_config = {
        #     heater: float(value) for heater, value in config.items()
        # }
        # print(printable_config)
        
        voltage_message = "".join(f"{heater},{value};" for heater, value in config.items()) + '\n'
        self.ser.write(voltage_message.encode())
        self.ser.flush()
        time.sleep(0.01)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

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
            h: float(np.clip(V_BIAS_INTERNAL + rng.normal(0, 2.5), V_MIN, V_MAX))
            for h in self.internal_heaters
        }


        #self.mesh_bias = {h: 0.0 for h in self.internal_heaters}
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
# PHOTONIC RESERVOIR
# ==========================

class PhotonicReservoirMNIST(PhotonicReservoir):
    """
    Photonic reservoir adapted for spatial pattern classification.
    Inherits from the time series version but modifies for spatial processing.
    """
    
    def __init__(self, input_heaters, all_heaters, scope_channels):
        super().__init__(input_heaters, all_heaters, scope_channels)
        print("[MNIST] Photonic reservoir initialized for spatial classification")
    
    def process_spatial_pattern(self, image_pixels):
        """
        Process a single image by chunking its pixels and scanning them sequentially.
        This method leverages the reservoir's temporal dynamics.
        """
        reservoir_features = []
        
        # The number of input heaters determines the chunk size for each time step.
        chunk_size = len(self.input_heaters)
        
        # Since the image is now 7x7 (49 pixels), the number of chunks is 49 // 7 = 7.
        num_chunks = len(image_pixels) // chunk_size
        
        # Iterate through all chunks of the resized image.
        for i in range(num_chunks):
            # Get the current chunk of pixels.
            chunk_start = i * chunk_size
            chunk_end = chunk_start + chunk_size
            pixel_chunk = image_pixels[chunk_start:chunk_end]
            
            # Map the pixel values to the input heaters.
            input_voltages = [
                V_MIN + (V_MAX - V_MIN) * pixel_value
                for pixel_value in pixel_chunk
            ]
            
            # Create a dictionary to hold the heater configuration.
            config = {
                heater: round(voltage, 2)
                for heater, voltage in zip(self.input_heaters, input_voltages)
            }
            
            # Send the voltage configuration to the heaters.
            self.bus.send(config)
            
            # Allow the system to settle and evolve.
            time.sleep(T_SETTLE)
            
            # Read the state of the reservoir from the photodetectors.
            pd_reading = self.scope.read_many(avg=READ_AVG)

            # Check for NaN immediately after scope read
            if np.any(np.isnan(pd_reading)):
                print(f"  NaN detected in chunk {i}: {pd_reading}")
                print(f"  Input voltages were: {input_voltages}")
                print(f"  Pixel chunk was: {pixel_chunk}")
                # Skip this entire image
                return np.full(num_chunks * len(self.scope.channels), np.nan)

            reservoir_features.append(pd_reading)

        # Flatten the list of readings into a single feature vector.
        # The length of this vector will be (7 chunks * 4 channels) = 28 features.
        return np.array(reservoir_features).flatten()

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

    classifiers = {
        'Logistic Regression': LogisticRegression(
            max_iter=10000,
            random_state=42,
            multi_class='multinomial',  # multinomial handles >2 classes properly
            solver='lbfgs'              # good for multinomial
        ),
        'Ridge Classifier': RidgeClassifier(alpha=1.0, random_state=42)
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
            from sklearn.pipeline import make_pipeline
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

        
        reservoir.close()

if __name__ == "__main__":  
    # For full MNIST classification
    main_mnist()