import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

# 1. Load MNIST and scale
mnist = fetch_openml("mnist_784", version=1)
X = mnist.data / 255.0
y = mnist.target.astype(int)

# 2. Split small subset for demo

X = X.to_numpy() / 255.0
y = y.to_numpy().astype(int)

idx = np.random.choice(len(X), 5000, replace=False)
X, y = X[idx], y[idx]

# 3. One-hot encode labels+
Y = np.eye(10)[y]

# 4. Random hidden layer (ELM)
n_hidden = 1000
W_in = np.random.normal(size=(X.shape[1], n_hidden))
b = np.random.normal(size=(n_hidden,))
H = np.tanh(X @ W_in + b)  # hidden activations

# 5. Solve output weights analytically (ridge regression)
ridge = Ridge(alpha=1e-2, fit_intercept=False)
ridge.fit(H, Y)
W_out = ridge.coef_.T

# 6. Test
Y_pred = H @ W_out
pred_labels = np.argmax(Y_pred, axis=1)
acc = np.mean(pred_labels == y)
print(f"ELM accuracy: {acc:.3f}")
