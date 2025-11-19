import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier, LinearRegression

# Load data
data = np.load('multi_lambda_20251118_205553.npz')
X = data["X_concat"]
y = data["y"]

# Training sizes to test
sizes = [100, 200, 300, 400,500, 600,650,700, 800, 850,  900]
ridge_acc = []
linreg_acc = []
linreg_acc_mesh = []

for n in sizes:
    # Stratified subsample
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=42
    )

    # Train/test split inside subsample
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.2, stratify=y_small, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ---- Fast Ridge (fixed alpha=100) ----
    ridge = RidgeClassifier(alpha=100.0)
    ridge.fit(X_train_s, y_train)
    y_pred = ridge.predict(X_test_s)
    ridge_acc.append(accuracy_score(y_test, y_pred))

    lin2 = LinearRegression()
    lin2.fit(X_train_s, y_train)
    y_pred_lin2 = np.rint(lin2.predict(X_test_s)).astype(int)
    y_pred_lin2 = np.clip(y_pred_lin2, 0, 9)
    linreg_acc_mesh.append(accuracy_score(y_test, y_pred_lin2))

    # ---- Linear Regression one-hot ----
    Y_train = np.eye(10)[y_train]
    lin = LinearRegression()
    lin.fit(X_train_s, Y_train)
    y_pred_lin = np.argmax(X_test_s @ lin.coef_.T + lin.intercept_, axis=1)
    linreg_acc.append(accuracy_score(y_test, y_pred_lin))

# ---- Plot ----
plt.figure(figsize=(6,4))
plt.plot(sizes, ridge_acc, 'o-', label='Ridge(alpha=100)')
plt.plot(sizes, linreg_acc_mesh, 'o-', label='Linear Regression (mesh features)')
plt.plot(sizes, linreg_acc, 'o-', label='Linear Regression (one-hot)')
plt.xlabel("Training samples")
plt.ylabel("Test accuracy")
plt.title("Learning Curve on Multi-Lambda Features")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
