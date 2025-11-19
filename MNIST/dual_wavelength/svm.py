import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load your file
data = np.load("multi_lambda_20251118_205553.npz")
X = data["X_concat"]
y = data["y"]

# Train/test split (same as your ELM)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# SVM with RBF kernel (the nonlinear one)
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_s, y_train)

# Evaluate
y_pred = svm.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print("SVM RBF accuracy:", acc)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=500, solver="lbfgs")
logreg.fit(X_train_s, y_train)
print("Logistic:", logreg.score(X_test_s, y_test))

from sklearn.svm import LinearSVC

lsvm = LinearSVC(C=1.0)
lsvm.fit(X_train_s, y_train)
print("Linear SVM:", lsvm.score(X_test_s, y_test))
