# photonic_esn_7wide_finalstate_feedback_v2.py
# ------------------------------------------------------------
# 7-wide photonic ESN with software feedback (final-state readout)
# Incorporates robustness/QoL updates from the MNIST spatial pipeline
# while preserving the ESN feedback core.
# ------------------------------------------------------------
import os
import time
from datetime import datetime
import numpy as np
from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- hardware I/O drivers (user-provided) ---
from Lib.scope import RigolDualScopes
from Lib.heater_bus import HeaterBus

# =========================================================
# CONFIG
# =========================================================
SCOPE1_CHANNELS = [1, 2, 3, 4]       # first scope (4 channels)
SCOPE2_CHANNELS = [1, 2, 3]          # second scope (3 channels)

INPUT_HEATERS       = (28, 29, 30, 31, 32, 33, 34)     # 7 inputs
ALL_HEATERS         = tuple(range(35))                  # includes internal mesh heaters

# Voltage rails and biases (keep generous headroom; avoid frequent clipping)
V_MIN, V_MAX        = 1.10, 4.50
V_BIAS_INPUT        = 2.50
V_BIAS_INTERNAL     = 2.50
MESH_NOISE_STD      = 0.40                               # random internal bias spread

# Pixel → voltage mapping around bias (centered pixels)
SPATIAL_GAIN        = 0.15                               # per-pixel swing around V_BIAS_INPUT
NOISE_LEVEL         = 0.02                               # small dither to break ties

# Timing / virtual nodes
T_SETTLE_ROW        = 0.08      # settle after driving one row
K_VIRTUAL           = 2         # quick reads per row (averaged); set 1 to disable
T_BETWEEN_VN        = 0.015
READ_AVG            = 1

# ESN (software feedback)
LEAK                = 0.6       # 0.3–0.8 typical
ALPHA               = 0.3       # recurrence strength (0 disables feedback)
GAMMA               = 1.0       # scale on normalized z
BIAS_TERM           = 0.0       # additive bias inside tanh

# Data
N_SAMPLES_PER_DIGIT = 150
TEST_FRACTION       = 0.2
SEED                = 42

# Output
MODEL_DIR           = Path("MNIST/models")
FIG_DIR             = Path("MNIST/figures")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# DATA LOADING (28x28 → 7x7 via 4x4 mean pooling)
# =========================================================
def load_mnist_resized_7x7(n_per_class=N_SAMPLES_PER_DIGIT, seed=SEED):
    """Balanced subset of MNIST, downsampled to 7x7 via 4x4 mean pooling.
    Returns X (N, 49 in [0,1]), y (N, int labels 0..9).
    """
    rng = np.random.default_rng(seed)
    try:
        X, y = fetch_openml(
            'mnist_784', version=1, as_frame=False, cache=True, parser='auto', return_X_y=True
        )
        y = y.astype(int)

        # Balanced sampling per class
        idxs = []
        for d in range(10):
            d_idx = np.where(y == d)[0]
            choose = min(n_per_class, len(d_idx))
            idxs.append(rng.choice(d_idx, size=choose, replace=False))
        idxs = np.concatenate(idxs)
        X, y = X[idxs], y[idxs]

        # Downsample 28x28 → 7x7 (mean pooling over 4x4)
        X = (X.reshape(-1, 28, 28) / 255.0).reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4))
        X = X.reshape(-1, 49)

        p = rng.permutation(len(X))
        return X[p], y[p]
    except Exception as e:
        print(f"[WARN] OpenML MNIST failed ({e}).")
        raise

# =========================================================
# UTILS
# =========================================================
def percent_clipped(cfg):
    vals = np.fromiter(cfg.values(), dtype=float)
    return 100.0 * ((vals <= V_MIN + 1e-12) | (vals >= V_MAX - 1e-12)).mean()

# =========================================================
# PHOTONIC ESN (7-at-a-time rows, final-state readout)
# =========================================================
class PhotonicESN7Wide:
    """
    - Drives 7 heaters per timestep with one row of a 7x7 image.
    - Reads PD vector z_t (n_channels=7).
    - State update: x_t = (1-leak) x_{t-1} + leak * tanh( GAMMA*z_norm + ALPHA*W_res x_{t-1} + bias ).
    - Uses FINAL STATE per image for classification.
    - Optional virtual nodes (K_VIRTUAL): multiple quick reads per row are averaged into z_t.
    """
    def __init__(self, input_heaters, all_heaters,
                 leak=LEAK, alpha=ALPHA, gamma=GAMMA, bias=BIAS_TERM, seed=SEED):
        self.rng = np.random.default_rng(seed)

        self.input_heaters    = list(input_heaters)
        self.internal_heaters = [h for h in all_heaters if h not in self.input_heaters]
        self.scope            = RigolDualScopes(SCOPE1_CHANNELS, SCOPE2_CHANNELS, serial_scope1='HDO1B244000779')
        self.bus              = HeaterBus()

        # Mesh bias with narrow spread (preserve headroom)
        self.mesh_bias = {
            h: float(np.clip(V_BIAS_INTERNAL + self.rng.normal(0.0, MESH_NOISE_STD), V_MIN, V_MAX))
            for h in self.internal_heaters
        }
        # Input bias
        self.input_bias = {h: V_BIAS_INPUT for h in self.input_heaters}

        # Apply initial config
        self.bus.send({**self.mesh_bias, **self.input_bias})
        time.sleep(0.2)

        # ESN params
        self.n_state = 7  # state dimension equals #scope channels
        self.x = np.zeros(self.n_state, dtype=float)

        self.leak  = float(leak)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.bias  = float(bias)

        # Random recurrent matrix for software feedback
        self.W_res = self.rng.normal(0.0, 1.0, size=(self.n_state, self.n_state)) / np.sqrt(self.n_state)

        # Running z-normalization (robust, momentum on L1-scale)
        self.z_mean  = np.zeros(self.n_state, dtype=float)
        self.z_scale = np.ones(self.n_state, dtype=float)
        self.z_mom   = 0.02  # momentum

        self.n_channels = self.n_state
        self.row_len    = len(self.input_heaters)

    def close(self):
        try: self.scope.close()
        except: pass
        try: self.bus.close()
        except: pass

    # ---------- hardware drive ----------
    def _drive_row(self, row_pixels):
        """Map 7 pixels in [0,1] to 7 voltages around bias; add tiny noise."""
        p = np.asarray(row_pixels, dtype=float) - 0.5            # center to [-0.5, +0.5]
        noise = self.rng.uniform(-1.0, 1.0, size=p.shape) * NOISE_LEVEL
        volts = V_BIAS_INPUT + SPATIAL_GAIN * (2.0 * p) + noise  # scale to ~[-1,1] range * gain
        volts = np.clip(volts, V_MIN, V_MAX)
        cfg = {h: float(v) for h, v in zip(self.input_heaters, volts)}
        self.bus.send(cfg)
        return cfg

    # ---------- ESN core ----------
    def _normalize_and_track(self, z):
        m, s, mom = self.z_mean, self.z_scale, self.z_mom
        m_new = (1 - mom) * m + mom * z
        s_new = (1 - mom) * s + mom * np.maximum(1e-3, np.abs(z - m_new))
        self.z_mean, self.z_scale = m_new, s_new
        return (z - m_new) / s_new

    def _step(self, z):
        """One ESN step with PD observation z (already read from hardware)."""
        z_norm = self._normalize_and_track(z)
        pre = self.gamma * z_norm + self.alpha * (self.W_res @ self.x) + self.bias
        x_new = np.tanh(pre)
        self.x = (1.0 - self.leak) * self.x + self.leak * x_new
        return self.x

    # ---------- public: process image / dataset ----------
    def reset_state(self):
        self.x[:] = 0.0

    def process_image_final_state(self, img_49):
        """
        Feeds a 7x7 image as 7 timesteps (rows). Returns the final ESN state (n_state,).
        Uses K_VIRTUAL quick reads per row and averages them as z_t.
        """
        x7 = np.asarray(img_49, dtype=float).reshape(7, 7)
        self.reset_state()
        avg_clip = 0.0

        for r in range(7):
            cfg = self._drive_row(x7[r])
            avg_clip += percent_clipped(cfg)

            time.sleep(T_SETTLE_ROW)

            # collect K_VIRTUAL quick reads and average (virtual nodes → one z_t)
            zs = []
            for k in range(K_VIRTUAL):
                if k > 0:
                    time.sleep(T_BETWEEN_VN)
                z = self.scope.read_many(avg=READ_AVG).astype(float)
                if np.any(np.isnan(z)):
                    return np.full(self.n_state, np.nan)
                if z.size != self.n_channels:
                    raise RuntimeError(f"Scope returned {z.size} channels, expected {self.n_channels}")
                zs.append(z)
            z_t = np.mean(np.vstack(zs), axis=0)

            # ESN update
            self._step(z_t)

        avg_clip /= 7.0
        if avg_clip > 1.0:
            print(f"[WARN] Avg clipped heaters this image: {avg_clip:.1f}% → consider lowering SPATIAL_GAIN or MESH_NOISE_STD.")
        return self.x.copy()

    def process_dataset(self, X49, y, phase="RUN"):
        """Returns final-state features Xs: [N_eff, n_state], labels ys"""
        print(f"[{phase}] Processing {len(X49)} images...")
        feats, ys = [], []
        for i, (img, yi) in enumerate(zip(X49, y)):
            if (i + 1) % 25 == 0:
                print(f"[{phase}] {i+1}/{len(X49)} done")
            try:
                xT = self.process_image_final_state(img)
                if not np.any(np.isnan(xT)):
                    feats.append(xT)
                    ys.append(yi)
                else:
                    print(f"[WARN] Skipping sample {i} (NaNs in state).")
            except Exception as e:
                print(f"[ERROR] Sample {i} failed: {e}")

        Xs = np.asarray(feats)
        ys = np.asarray(ys)
        print(f"[{phase}] Completed: {Xs.shape[0]} samples, feature dim: {Xs.shape[1] if Xs.size else 0}")
        return Xs, ys

# =========================================================
# TRAIN / EVAL
# =========================================================
def train_final_state_classifier(Xs, ys):
    """Train simple linear readouts on final states with scaling + CV reporting."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, ys, test_size=TEST_FRACTION, stratify=ys, random_state=SEED
    )

    models = {
        "LogReg": LogisticRegression(max_iter=5000, multi_class="multinomial", solver="lbfgs", random_state=SEED),
        "RidgeC": RidgeClassifier(alpha=1.0, random_state=SEED),
    }

    results = {}
    for name, clf in models.items():
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc_train = pipe.score(X_tr, y_tr)
        acc_test  = accuracy_score(y_te, y_pred)
        cv = cross_val_score(pipe, Xs, ys, cv=5)

        print(f"[{name}] Train {acc_train:.3f} | Test {acc_test:.3f} | CV {cv.mean():.3f} ± {cv.std():.3f}")
        results[name] = dict(pipe=pipe, y_pred=y_pred, y_test=y_te, acc_test=acc_test)

    # pick best by test accuracy
    best_name, best_res = max(results.items(), key=lambda kv: kv[1]["acc_test"])  # (name, res)
    print(f"Best model: {best_name} (Test acc: {best_res['acc_test']:.3f})")
    print("Classification report:")
    print(classification_report(best_res["y_test"], best_res["y_pred"]))
    print("Confusion matrix:")
    print(confusion_matrix(best_res["y_test"], best_res["y_pred"]))

    return best_res["pipe"], results

# =========================================================
# VISUALIZATION (match the non-feedback script style)
# =========================================================

def visualize_results(results, *, total_seconds=None, run_tag=None):
    """Replicates the figure style from the non-feedback MNIST script.
    - Left: Confusion matrix (uses Logistic Regression predictions if available; else best model)
    - Right: Per-digit accuracy bar chart (from the same predictions)
    - Title: includes meta like K_VIRTUAL, SPATIAL_GAIN, READ_AVG, LEAK, ALPHA, GAMMA
    Saves PNG to MNIST/figures/.
    """
    import matplotlib.pyplot as plt
    from datetime import datetime
    os.makedirs(FIG_DIR, exist_ok=True)

    # Prefer Logistic Regression to mirror the other script; fallback to best
    if "LogReg" in results:
        pred = results["LogReg"]["y_pred"]
        y_test = results["LogReg"]["y_test"]
        used_name = "Logistic Regression"
        used_acc = results["LogReg"]["acc_test"]
    else:
        best_name, best_res = max(results.items(), key=lambda kv: kv[1]["acc_test"])  # (name, res)
        pred = best_res["y_pred"]; y_test = best_res["y_test"]
        used_name = best_name; used_acc = best_res["acc_test"]

    plt.figure(figsize=(12, 4))

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, pred)
    plt.subplot(1, 2, 1)
    plt.imshow(cm, interpolation='nearest')  # default colormap (no explicit colors)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Per-digit accuracy
    plt.subplot(1, 2, 2)
    digit_accuracies = []
    for digit in range(10):
        mask = (y_test == digit)
        if np.sum(mask) > 0:
            acc = accuracy_score(y_test[mask], pred[mask])
        else:
            acc = 0.0
        digit_accuracies.append(acc)
    plt.bar(range(10), digit_accuracies)
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.title('Per-Digit Accuracy')
    plt.xticks(range(10))

    # Title with metadata, matching the other script's style
    time_str = f"{total_seconds:.2f}s" if isinstance(total_seconds, (int, float)) else "N/A"
    meta_line_1 = f"Best (plotted): {used_name}  |  Test Acc: {used_acc:.3f}  |  Time: {time_str}"
    meta_line_2 = (
        f"K={K_VIRTUAL}, SPATIAL_GAIN={SPATIAL_GAIN}, READ_AVG={READ_AVG}, "
        f"LEAK={LEAK}, ALPHA={ALPHA}, GAMMA={GAMMA}"
    )
    plt.suptitle(meta_line_1 + "" + meta_line_2, y=1.05, fontsize=11)
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    fname_base = f"mnist_photonic_results{tag}_{ts}.png"
    png_path = FIG_DIR / fname_base
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"[Saved] {png_path}")

# =========================================================
# MAIN
# =========================================================
def main():
    np.random.seed(SEED)
    print("="*72)
    print(" PHOTONIC ESN — 7-at-a-time rows, software feedback, final-state readout ")
    print("="*72)

    # Load data
    X49, y = load_mnist_resized_7x7(N_SAMPLES_PER_DIGIT, SEED)
    print(f"Dataset: {len(X49)} samples, classes: {sorted(set(y))}")

    esn = PhotonicESN7Wide(INPUT_HEATERS, ALL_HEATERS,
                           leak=LEAK, alpha=ALPHA, gamma=GAMMA, bias=BIAS_TERM, seed=SEED)
    try:
        # Generate final-state features from hardware
        Xs, ys = esn.process_dataset(X49, y, phase="MNIST-ESN")

        if Xs.shape[0] < 5:
            print("Too few valid samples — abort.")
            return

        # Train readout on final states
        best_model, results = train_final_state_classifier(Xs, ys)

        # Rough runtime per image
        per_img_time = 7 * (T_SETTLE_ROW + max(0, K_VIRTUAL - 1) * T_BETWEEN_VN)
        best_test = max(r['acc_test'] for r in results.values())

        # --- Visualization to match non-feedback script ---
        t0 = time.perf_counter()
        visualize_results(results, total_seconds=per_img_time * len(ys), run_tag="esn_finalstate")

        # Save model bundle
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = MODEL_DIR / f"photonic_esn7_finalstate_{ts}.npz"
        np.savez_compressed(
            save_path,
            model=None,  # placeholder: use pickle if you need to persist the sklearn pipeline
            meta=dict(
                SEED=SEED,
                LEAK=LEAK, ALPHA=ALPHA, GAMMA=GAMMA, BIAS_TERM=BIAS_TERM,
                SPATIAL_GAIN=SPATIAL_GAIN, NOISE_LEVEL=NOISE_LEVEL,
                T_SETTLE_ROW=T_SETTLE_ROW, K_VIRTUAL=K_VIRTUAL, T_BETWEEN_VN=T_BETWEEN_VN,
                READ_AVG=READ_AVG,
                N_SAMPLES_PER_DIGIT=N_SAMPLES_PER_DIGIT,
                TEST_FRACTION=TEST_FRACTION,
                best_test=best_test,
            ),
            Xs=Xs, ys=ys
        )

        print("" + "="*72)
        print(" SUMMARY ")
        print("="*72)
        print(f"Final-state dim: {Xs.shape[1]} = #scope channels ({7})")
        print(f"Approx time/image: ~{per_img_time:.2f}s")
        print(f"Best test accuracy: {best_test:.3f}")
        print(f"Saved run to: {save_path}")
        print("="*72)
        print(f"Final-state dim: {Xs.shape[1]} = #scope channels ({7})")
        print(f"Approx time/image: ~{per_img_time:.2f}s")
        print(f"Best test accuracy: {best_test:.3f}")
        print(f"Saved run to: {save_path}")
        print("="*72)

    finally:
        esn.close()


if __name__ == "__main__":
    main()
