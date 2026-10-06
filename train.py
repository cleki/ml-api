import os
import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

os.makedirs("model", exist_ok=True)
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, "model/model.joblib")
print("Saved. Pin this in requirements.txt -> scikit-learn==" + sklearn.__version__)