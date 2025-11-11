# iris_hw_one_heater_per_feature.py
import os
import time
from datetime import datetime
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeClassifierCV, LogisticRegression

# ---- Your hardware libs ----
from Lib.scope import RigolDualScopes
from Lib.DualBoard import DualAD5380Controller

# ==========================
# CONFIG (IRIS + Hardware)
# ==========================
FEATURE_STORE = os.path.join("IRIS", "feature_store_one2one.npz")

# Scopes / channels (unchanged)
SCOPE1_CHANNELS = [1, 2, 3, 4]   # first scope (4 channels)
SCOPE2_CHANNELS = [1, 2, 3]      # second scope (3 channels)

# Full input heater bank on your board
ALL_INPUT_HEATERS = [28, 29, 30, 31, 32, 33, 34]
ALL_HEATERS       = list(range(35))

# ==== Choose 3 or 4 features and map 1:1 to heaters ====
# Iris features: 0=sepal L, 1=sepal W, 2=petal L, 3=petal W
FEATURE_IDX = [0, 2, 3]            # <-- 3 features (strong trio)
#FEATURE_IDX = [0, 1, 2, 3]       # <-- uncomment to use all 4 features

# Pick the same number of heaters as features (keep order!)
ACTIVE_INPUT_HEATERS = [28, 29, 30]   # for 3 features
#ACTIVE_INPUT_HEATERS = [28, 29, 30, 31]  # for 4 features

assert len(FEATURE_IDX) == len(ACTIVE_INPUT_HEATERS), "FEATURE_IDX and ACTIVE_INPUT_HEATERS must match in length"

# Voltage limits / biases
V_MIN, V_MAX     = 1.10, 4.90
V_BIAS_INPUT     = 3.40

# Virtual masking (not needed here; keep 1)
K_VIRTUAL        = 1         # 1 = no masks; >1 adds ±1 masks over ACTIVE heaters
READ_AVG         = 1
GAIN             = 0.6       # keep <= ~0.64 to avoid clipping

# Dataset / targets
N_SAMPLES_PER_CLASS  = 50    # up to ~50 per class in IRIS
TEST_FRACTION        = 0.2
SEED                 = 42

# ==========================
# Feature store helpers
# ==========================
def save_feature_store(X, y, path=FEATURE_STORE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))
    print(f"[FeatureStore] Saved {len(y)} samples to {path}")

def load_feature_store(path=FEATURE_STORE):
    if os.path.exists(path):
        d = np.load(path)
        return d["X"], d["y"]
    return None, None

def append_feature_store(X_new, y_new, path=FEATURE_STORE):
    X_old, y_old = load_feature_store(path)
    if X_old is None:
        X_all, y_all = X_new, y_new
    else:
        X_all = np.concatenate([X_old, X_new], axis=0)
        y_all = np.concatenate([y_old, y_new], axis=0)
    #save_feature_store(X_all, y_all, path)
    return X_all, y_all

def per_class_counts(y):
    if y is None:
        return None
    labels, counts = np.unique(y, return_counts=True)
    out = np.zeros(labels.max() + 1, dtype=int)
    out[labels] = counts
    return out

# ==========================
# Hardware reservoir (one-to-one)
# ==========================
class PhotonicReservoirIrisOne2One:
    """
    Maps each normalized feature x_i in [0,1] to exactly one heater.
    No projection. Optionally applies ±1 masks across the ACTIVE heaters (K_VIRTUAL > 1).
    """
    def __init__(self, active_input_heaters, all_heaters):
        self.active = list(active_input_heaters)                    # N_active (3 or 4)
        self.idle_inputs = [h for h in ALL_INPUT_HEATERS if h not in self.active]
        self.internal = [h for h in all_heaters if h not in ALL_INPUT_HEATERS]

        self.scope = RigolDualScopes(SCOPE1_CHANNELS, SCOPE2_CHANNELS, serial_scope1='HDO1B244000779')
        self.bus   = DualAD5380Controller()

        rng = np.random.default_rng(42)   # or remove seed for different randomness each run

        # Random mesh biases uniformly in [V_MIN, V_MAX]
        self.mesh_bias = {
            h: float(rng.uniform(V_MIN, V_MAX))
            for h in self.internal
        }

        self.idle_bias  = {h: V_BIAS_INPUT for h in self.idle_inputs}
        self.base_bias  = {h: V_BIAS_INPUT for h in self.active}

        # Masks (over ACTIVE heaters only). Keep 1 (no masks) by default.
        if K_VIRTUAL <= 1:
            self.mask_matrix = np.ones((1, len(self.active)), dtype=float)
        else:
            rng = np.random.default_rng(42)
            micro = []
            while len(micro) < (K_VIRTUAL - 1):
                v = rng.choice([-1.0, 1.0], size=len(self.active))
                # simple orthogonality check
                if all(abs(np.dot(v, m) / len(self.active)) < 0.2 for m in micro):
                    micro.append(v)
            self.mask_matrix = np.vstack([np.ones((1, len(self.active))), np.array(micro, float)])

        # Apply initial baseline (mesh + idle + base; later keys override earlier ones)
        baseline = {**self.mesh_bias, **self.idle_bias, **self.base_bias}

        # Deterministic order → lists
        chs = sorted(baseline.keys())
        vs  = [baseline[h] for h in chs]
        time.sleep(0.02)
        # Direct hardware call (no HeaterBus)
        self.bus.set(chs, vs)
        # (Optional) keep your debug prints
        print(self.mesh_bias)
        print(self.base_bias)


    def close(self):

        self.scope.close()

            #self.bus.close()

    def process_vector(self, x_norm01):
        """
        x_norm01: shape (N_active,) in [0,1].
        For each mask (K_VIRTUAL times), set active heaters to:
            v = V_BIAS_INPUT + GAIN * 0.5*(V_MAX-V_MIN) * ((x-0.5)*2) * mask
        and read the scope. Features are concatenated across masks.
        """
        x = np.asarray(x_norm01, float).ravel()
        assert x.size == len(self.active), f"Expected {len(self.active)} inputs, got {x.size}"
        x_ctr = (x - 0.5) * 2.0
        half = 0.5 * (V_MAX - V_MIN)

        features = []
        for m in self.mask_matrix:
            v_act = V_BIAS_INPUT + GAIN * half * (x_ctr * m)              # per active heater
            v_act = np.clip(v_act, V_MIN, V_MAX)
            # Build command: active -> v_act; idle inputs held at bias; mesh at mesh_bias
            cmd = {h: float(vv) for h, vv in zip(self.active, v_act)}
            chs = sorted(cmd.keys())
            vs  = [cmd[h] for h in chs]
            print(chs,vs)
            self.bus.set(chs, vs)
            pd = self.scope.read_many(avg=int(READ_AVG))    # returns detector vector
            features.append(pd)

        return np.asarray(features, float).ravel()

    def process_dataset(self, X_norm01, y, phase_name="TRAINING",
                        existing_counts=None, target_per_class=None):
        print(f"[{phase_name}] Candidate samples: {len(X_norm01)}")
        n_classes = int(np.max(y)) + 1
        need = int(target_per_class) if target_per_class is not None else None
        have = np.zeros(n_classes, dtype=int) if existing_counts is None else existing_counts.copy()

        if need is not None and np.all(have >= need):
            print(f"[{phase_name}] Already have ≥{need} per class — no measurement.")
            return np.empty((0,)), np.empty((0,), dtype=int)

        X_new, y_new = [], []
        missing = None if need is None else np.maximum(need - have, 0)

        for i, (xv, lab) in enumerate(zip(X_norm01, y)):
            if (i + 1) % 25 == 0:
                print(f"[{phase_name}] Processed {i+1}/{len(X_norm01)}")

            if need is not None:
                if missing[lab] <= 0:
                    continue
                if np.all(missing <= 0):
                    break

        
            feats = self.process_vector(xv)
            if not np.any(np.isnan(feats)):
                X_new.append(feats)
                y_new.append(lab)
                if need is not None:
                    have[lab] += 1
                    missing[lab] -= 1
            else:
                print(f"[WARN] NaN features at idx {i}; skipped.")


        if len(X_new) == 0:
            print(f"[{phase_name}] No new samples measured.")
            return np.empty((0,)), np.empty((0,), dtype=int)

        X_new = np.asarray(X_new, float)
        y_new = np.asarray(y_new,  int)
        print(f"[{phase_name}] Measured new: {len(y_new)} | approx per-class: {have}")
        return X_new, y_new

# ==========================
# Training heads (same as before)
# ==========================
def train_heads(X_features, y_labels, *, seed=42):
    print("\n[ELM] Training one-layer heads on fixed features…")
    print(f"[ELM] Feature dim: {X_features.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=TEST_FRACTION,
        stratify=y_labels, random_state=seed
    )
    print(f"[ELM] Train: {len(X_train)} | Test: {len(X_test)}")

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    classes = np.unique(y_labels)
    n_classes = classes.size
    Y_train = np.eye(n_classes)[y_train]

    models, results = {}, {}

    # OLS
    linreg = LinearRegression()
    linreg.fit(X_train_s, Y_train)
    y_pred_tr = np.argmax(X_train_s @ linreg.coef_.T + linreg.intercept_, axis=1)
    y_pred_te = np.argmax(X_test_s  @ linreg.coef_.T + linreg.intercept_, axis=1)
    acc_tr, acc_te = accuracy_score(y_train, y_pred_tr), accuracy_score(y_test, y_pred_te)
    print(f"\n[LINREG] Train: {acc_tr:.3f} | Test: {acc_te:.3f}")
    print(classification_report(y_test, y_pred_te, zero_division=0))
    print(confusion_matrix(y_test, y_pred_te))
    models["linreg"] = linreg
    results["linreg"] = {"train_accuracy": acc_tr, "test_accuracy": acc_te, "y_pred": y_pred_te}

    # Ridge
    ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13))
    ridge.fit(X_train_s, y_train)
    y_pred_tr2, y_pred_te2 = ridge.predict(X_train_s), ridge.predict(X_test_s)
    acc_tr2, acc_te2 = accuracy_score(y_train, y_pred_tr2), accuracy_score(y_test, y_pred_te2)
    print(f"\n[RIDGE-CLF] Train: {acc_tr2:.3f} | Test: {acc_te2:.3f} | alpha={ridge.alpha_}")
    print(classification_report(y_test, y_pred_te2, zero_division=0))
    print(confusion_matrix(y_test, y_pred_te2))
    models["ridge_clf"] = ridge
    results["ridge_clf"] = {"train_accuracy": acc_tr2, "test_accuracy": acc_te2, "y_pred": y_pred_te2, "alpha": ridge.alpha_}

    # Logistic
    logreg = LogisticRegression(max_iter=20000, solver="lbfgs", C=1.0, multi_class="auto")
    logreg.fit(X_train_s, y_train)
    y_pred_tr3, y_pred_te3 = logreg.predict(X_train_s), logreg.predict(X_test_s)
    acc_tr3, acc_te3 = accuracy_score(y_train, y_pred_tr3), accuracy_score(y_test, y_pred_te3)
    print(f"\n[LOGREG]  Train: {acc_tr3:.3f} | Test: {acc_te3:.3f}")
    print(classification_report(y_test, y_pred_te3, zero_division=0))
    print(confusion_matrix(y_test, y_pred_te3))
    models["logreg"] = logreg
    results["logreg"] = {"train_accuracy": acc_tr3, "test_accuracy": acc_te3, "y_pred": y_pred_te3}

    results["scaler"] = scaler
    return models, results, (X_test, y_test)

# ==========================
# Visualization
# ==========================
def visualize_results(y_test, results, *, total_seconds=None, run_tag=None):
    os.makedirs(os.path.join("IRIS", "figures"), exist_ok=True)
    # pick best
    cands = [(k, v) for k, v in results.items() if isinstance(v, dict) and "test_accuracy" in v]
    best_name, best_info = max(cands, key=lambda kv: kv[1]["test_accuracy"])
    y_pred = np.asarray(best_info["y_pred"])
    classes_sorted = np.unique(y_test)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred, labels=classes_sorted)
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion Matrix ({best_name})')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(len(classes_sorted)), classes_sorted)
    plt.yticks(range(len(classes_sorted)), classes_sorted)

    plt.subplot(1, 2, 2)
    per_cls = []
    for c in classes_sorted:
        m = (y_test == c)
        per_cls.append(accuracy_score(y_test[m], y_pred[m]))
    plt.bar(range(len(classes_sorted)), per_cls)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title(f'Per-Class Accuracy ({best_name})')
    plt.xticks(range(len(classes_sorted)), classes_sorted)

    tstr = f"{total_seconds:.2f}s" if isinstance(total_seconds,(int,float)) else "N/A"
    plt.suptitle(f"Best: {best_name} | Test Acc: {best_info['test_accuracy']:.3f} | Time: {tstr}", y=1.05)
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join("IRIS", "figures", f"iris_one2one_{run_tag or 'run'}_{ts}.png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"[Saved] {png_path}")

# ==========================
# MAIN
# ==========================
def main():
    print("="*60)
    print(f"IRIS + PHOTONIC (one feature ↔ one heater)  |  features: {FEATURE_IDX}  |  heaters: {ACTIVE_INPUT_HEATERS}")
    print("="*60)
    t0 = time.perf_counter()
    reservoir = None

    try:
        # Load & select features
        iris = load_iris()
        X_full = iris.data.astype(np.float32)
        y_full = iris.target.astype(int)
        X_sel  = X_full[:, FEATURE_IDX]

        # Normalize each feature to [0,1] for hardware drive
        scaler01 = MinMaxScaler()
        X_01 = scaler01.fit_transform(X_sel)

        # Fast path: cache check
        X_cached, y_cached = load_feature_store(FEATURE_STORE)
        target = N_SAMPLES_PER_CLASS
        if X_cached is not None:
            counts = per_class_counts(y_cached)
            if counts is not None and np.all(counts >= target):
                print(f"[FastPath] Cache already has ≥{target} per class. Skipping measurement.")
                Xb, yb = _balanced_slice(X_cached, y_cached, target, classes=np.unique(y_full))
                models, results, test_data = train_heads(Xb, yb)
                visualize_results(test_data[1], results, total_seconds=time.perf_counter()-t0, run_tag="cached")
                _save(models, results)
                return

        counts = per_class_counts(y_cached) if y_cached is not None else np.zeros(len(np.unique(y_full)), dtype=int)

        # Measure only what's missing
        reservoir = PhotonicReservoirIrisOne2One(ACTIVE_INPUT_HEATERS, ALL_HEATERS)
        X_new, y_new = reservoir.process_dataset(
            X_01, y_full, phase_name="TRAINING",
            existing_counts=counts, target_per_class=target
        )

        if reservoir is not None:
            reservoir.close(); reservoir = None

        if X_new.size > 0:
            X_all, y_all = append_feature_store(X_new, y_new, FEATURE_STORE)
        else:
            X_all, y_all = (X_cached, y_cached)

        counts_after = per_class_counts(y_all)
        if counts_after is None or np.any(counts_after < target):
            print(f"[WARN] Still short per class: {counts_after} (need {target}). Aborting training.")
            return

        Xb, yb = _balanced_slice(X_all, y_all, target, classes=np.unique(y_full))
        models, results, (X_test, y_test) = train_heads(Xb, yb)

        visualize_results(y_test, results, total_seconds=time.perf_counter()-t0, run_tag="hardware")
        _save(models, results)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        if reservoir is not None:
            try:
                reservoir.close()
            except Exception:
                pass

def _balanced_slice(X, y, n_per_class, *, classes):
    rng = np.random.default_rng(SEED)
    Xb, yb = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        take = rng.choice(idx, size=n_per_class, replace=False)
        Xb.append(X[take]); yb.append(y[take])
    return np.vstack(Xb), np.concatenate(yb)

def _save(models, results):
    os.makedirs(os.path.join("IRIS", "models"), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("IRIS", "models", f"iris_one2one_{ts}.pkl")
    with open(path, 'wb') as f:
        pickle.dump({
            'models': models,
            'results': results,
            'config': {
                'FEATURE_IDX': FEATURE_IDX,
                'ACTIVE_INPUT_HEATERS': ACTIVE_INPUT_HEATERS,
                'V_MIN': V_MIN, 'V_MAX': V_MAX,
                'V_BIAS_INPUT': V_BIAS_INPUT,
                'GAIN': GAIN, 'K_VIRTUAL': K_VIRTUAL
            }
        }, f)
    print(f"Classifier saved to '{path}'")

if __name__ == "__main__":
    main()
