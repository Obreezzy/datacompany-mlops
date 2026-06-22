import pandas as pd
import numpy as np
import joblib
import json
import os
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

print("DataCompany MLOps — Model Training Pipeline")
print("="*55)

# ============================================================
# STEP 1 - Generate training data
# ============================================================

print("\nStep 1: Loading training data...")

np.random.seed(42)
N = 1000

df = pd.DataFrame({
    "missed_payments":     np.random.poisson(1.5, N),
    "inspection_score":    np.round(np.random.uniform(3, 10, N), 1),
    "complaints_received": np.random.poisson(0.5, N),
    "tenancy_months":      np.random.randint(1, 120, N),
    "maintenance_calls":   np.random.poisson(2, N),
    "property_size":       np.random.choice([1,2,3,4], N, p=[0.3,0.4,0.2,0.1]),
})

risk_score = (
    df["missed_payments"] * 2.5 +
    df["complaints_received"] * 1.5 +
    (10 - df["inspection_score"]) * 0.8 +
    df["maintenance_calls"] * 0.3 +
    np.random.normal(0, 0.5, N)
)

df["risk_label"] = pd.cut(
    risk_score,
    bins=[-np.inf, 4, 8, np.inf],
    labels=["LOW", "MEDIUM", "HIGH"]
)

le = LabelEncoder()
df["risk_encoded"] = le.fit_transform(df["risk_label"])

feature_cols = [
    "missed_payments",
    "inspection_score",
    "complaints_received",
    "tenancy_months",
    "maintenance_calls",
    "property_size"
]

X = df[feature_cols]
y = df["risk_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples:  {len(X_test)}")

# ============================================================
# STEP 2 - Train model
# ============================================================

print("\nStep 2: Training model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("  Model trained.")

# ============================================================
# STEP 3 - Evaluate
# ============================================================

print("\nStep 3: Evaluating model...")

y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average="weighted")

print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================================================
# STEP 4 - Quality gate
# ============================================================

print("\nStep 4: Quality gate check...")

ACCURACY_THRESHOLD = 0.75
F1_THRESHOLD       = 0.70

if accuracy < ACCURACY_THRESHOLD:
    print(f"  FAILED: Accuracy {accuracy:.4f} below threshold {ACCURACY_THRESHOLD}")
    exit(1)

if f1 < F1_THRESHOLD:
    print(f"  FAILED: F1 score {f1:.4f} below threshold {F1_THRESHOLD}")
    exit(1)

print(f"  PASSED: Accuracy {accuracy:.4f} >= {ACCURACY_THRESHOLD}")
print(f"  PASSED: F1 Score {f1:.4f} >= {F1_THRESHOLD}")

# ============================================================
# STEP 5 - Version and save
# ============================================================

print("\nStep 5: Versioning and saving model...")

version    = datetime.now().strftime("v%Y%m%d_%H%M%S")
model_dir  = f"model_registry/{version}"
os.makedirs(model_dir, exist_ok=True)

# Save versioned model
joblib.dump(model, f"{model_dir}/risk_model.pkl")
joblib.dump(le,    f"{model_dir}/label_encoder.pkl")

metadata = {
    "version":          version,
    "accuracy":         round(accuracy, 4),
    "f1_score":         round(f1, 4),
    "training_samples": len(X_train),
    "features":         feature_cols,
    "model_type":       "RandomForestClassifier",
    "n_estimators":     100,
    "trained_at":       datetime.now().isoformat(),
    "classes":          list(le.classes_),
    "status":           "candidate"   # candidate → production
}

with open(f"{model_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ============================================================
# STEP 6 - Update version registry
# Keeps a log of every model ever trained
# ============================================================

print("\nStep 6: Updating version registry...")

registry_path = "model_registry/registry.json"

if os.path.exists(registry_path):
    with open(registry_path) as f:
        registry = json.load(f)
else:
    registry = {"models": [], "production": None}

registry["models"].append({
    "version":   version,
    "accuracy":  round(accuracy, 4),
    "f1_score":  round(f1, 4),
    "trained_at": datetime.now().isoformat(),
    "status":    "candidate"
})

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

# ============================================================
# STEP 7 - Copy to active model/  folder for the API
# ============================================================

print("\nStep 7: Promoting to active model...")

os.makedirs("model", exist_ok=True)
shutil.copy(f"{model_dir}/risk_model.pkl",   "model/risk_model.pkl")
shutil.copy(f"{model_dir}/label_encoder.pkl","model/label_encoder.pkl")
shutil.copy(f"{model_dir}/metadata.json",    "model/metadata.json")

# Mark as production in registry
registry["models"][-1]["status"] = "production"
registry["production"] = version

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

print(f"\n  Version:    {version}")
print(f"  Saved to:   {model_dir}/")
print(f"  Active:     model/")
print(f"  Status:     production")
print("\nTraining pipeline complete.")