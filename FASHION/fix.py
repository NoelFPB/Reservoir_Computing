import numpy as np
import os

# ----------------------------------------------------------
# Path to your corrupted dataset
# ----------------------------------------------------------
PATH = "dual_wavelength/multi_lambda_20251119_162719.npz"

# Output file (safe – does not overwrite original)
FIXED_PATH = PATH.replace(".npz", "_FIXED.npz")

# ----------------------------------------------------------
# Load and inspect
# ----------------------------------------------------------
print(f"[LOAD] Loading: {PATH}")
d = np.load(PATH)

X_stack   = d["X_stack"]
X_concat  = d["X_concat"]
y         = d["y"]
waves     = d["wavelengths"]

N = X_stack.shape[0]
Ny = len(y)

print(f"X_stack shape   = {X_stack.shape}")
print(f"X_concat shape  = {X_concat.shape}")
print(f"y shape         = {y.shape}")
print(f"wavelengths     = {waves}")

# ----------------------------------------------------------
# Detect mismatch
# ----------------------------------------------------------
if Ny == N:
    print("[OK] Dataset is already consistent. Nothing to repair.")
    exit()

elif Ny < N:
    print(f"[ERROR] Labels fewer than features: Ny={Ny}, N={N}")
    print("Cannot repair safely. You must regenerate the dataset.")
    exit()

else:
    print(f"[FIX] Trimming y from {Ny} → {N}")
    y_fixed = y[:N]

# ----------------------------------------------------------
# Sanity check: class balance
# ----------------------------------------------------------
unique, counts = np.unique(y_fixed, return_counts=True)
print("\n[CHECK] Label distribution after trimming:")
for u, c in zip(unique, counts):
    print(f"  class {u}: {c}")

# Optional: warn if imbalance looks suspicious
if counts.min() < counts.max() - 5:
    print("\n[WARN] The class distribution looks uneven!")
    print("       This might indicate deeper corruption.\n")

# ----------------------------------------------------------
# Save the repaired file
# ----------------------------------------------------------
np.savez_compressed(
    FIXED_PATH,
    X_stack=X_stack,
    X_concat=X_concat,
    y=y_fixed,
    wavelengths=waves,
)

print(f"\n[SAVE] Repaired dataset written to:")
print(f"       {os.path.abspath(FIXED_PATH)}")
print("\nYou can now use this file safely.\n")
