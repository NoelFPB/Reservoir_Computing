import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --------------------------
# Load your multi-λ dataset
# --------------------------
data = np.load("multi_lambda_20251120_011637.npz")
X_stack = data["X_stack"]         # (N, L, D)
y = data["y"]                     # (N,)
wavelengths = data["wavelengths"] # shape (L,)

N, L, D = X_stack.shape
print("Shape:", X_stack.shape)

# ----------------------------------------
# Choose number of wavelengths to visualize
# ----------------------------------------
# If you want all wavelengths:
# wavelengths_to_use = range(L)
# If you want specific wavelengths:
wavelengths_to_use = range(L)

# ----------------------------------------
# Prepare figure
# ----------------------------------------
plt.figure(figsize=(14, 4 * (len(wavelengths_to_use)+1)))

# ----------------------------------------
# Plot PCA for each wavelength separately
# ----------------------------------------
for i in wavelengths_to_use:
    X_i = X_stack[:, i, :]  # (N, D)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_i)

    plt.subplot(len(wavelengths_to_use)+1, 1, i+1)
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='tab10', s=8)
    plt.title(f"PCA of wavelength λ={wavelengths[i]} nm")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

# ----------------------------------------
# PCA of concatenated wavelengths
# ----------------------------------------
X_concat = data["X_concat"]  # (N, L*D)

pca = PCA(n_components=2)
X_pca_concat = pca.fit_transform(X_concat)

plt.subplot(len(wavelengths_to_use)+1, 1, len(wavelengths_to_use)+1)
plt.scatter(X_pca_concat[:,0], X_pca_concat[:,1], c=y, cmap='tab10', s=8)
plt.title("PCA of concatenated multi-wavelength features")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()
