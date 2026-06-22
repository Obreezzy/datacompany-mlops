"""
monitor.py — DataCompany Model Monitoring

Simulates production traffic and detects:
  1. Data drift       — feature distributions shifting
  2. Prediction drift — output label distribution shifting
  3. Performance drift — accuracy dropping on labelled samples

Run after deployment to check model health.
"""

import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime
from collections import Counter

print("DataCompany MLOps — Model Monitor")
print("="*55)

# ============================================================
# Load model
# ============================================================

model   = joblib.load("model/risk_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")

with open("model/metadata.json") as f:
    metadata = json.load(f)

print(f"\nMonitoring model version: {metadata['version']}")
print(f"Trained accuracy: {metadata['accuracy']}")

FEATURE_COLS = metadata["features"]

# ============================================================
# BASELINE — what training data looked like
# This is what the model expects to see
# ============================================================

baseline = {
    "missed_payments":     {"mean": 1.5,  "std": 1.2},
    "inspection_score":    {"mean": 6.5,  "std": 2.0},
    "complaints_received": {"mean": 0.5,  "std": 0.7},
    "tenancy_months":      {"mean": 60.0, "std": 35.0},
    "maintenance_calls":   {"mean": 2.0,  "std": 1.4},
    "property_size":       {"mean": 2.1,  "std": 0.9},
}

baseline_label_dist = {"LOW": 0.45, "MEDIUM": 0.35, "HIGH": 0.20}

# ============================================================
# SIMULATE production traffic — 3 scenarios
# ============================================================

np.random.seed(99)

def simulate_requests(n, drift=False, drift_feature=None, drift_amount=0):
    """Generate simulated incoming tenant prediction requests."""
    data = {
        "missed_payments":     np.random.poisson(1.5, n),
        "inspection_score":    np.round(np.random.uniform(3, 10, n), 1),
        "complaints_received": np.random.poisson(0.5, n),
        "tenancy_months":      np.random.randint(1, 120, n),
        "maintenance_calls":   np.random.poisson(2, n),
        "property_size":       np.random.choice([1,2,3,4], n, p=[0.3,0.4,0.2,0.1]),
    }
    if drift and drift_feature:
        data[drift_feature] = data[drift_feature] + drift_amount
    return pd.DataFrame(data)

# ============================================================
# CHECK 1 — Data drift detection
# Compare incoming feature means to baseline
# Alert if mean shifts by more than 2 standard deviations
# ============================================================

print("\n" + "="*55)
print("CHECK 1: Data Drift Detection")
print("="*55)

# Simulate normal traffic
normal_traffic = simulate_requests(200)

drift_alerts = []

for feature in FEATURE_COLS:
    incoming_mean = normal_traffic[feature].mean()
    baseline_mean = baseline[feature]["mean"]
    baseline_std  = baseline[feature]["std"]

    deviation = abs(incoming_mean - baseline_mean) / baseline_std

    status = "OK" if deviation < 2.0 else "DRIFT DETECTED"
    if deviation >= 2.0:
        drift_alerts.append(feature)

    print(f"  {feature:<25} baseline={baseline_mean:.2f}  "
          f"incoming={incoming_mean:.2f}  "
          f"deviation={deviation:.2f}σ  [{status}]")

if not drift_alerts:
    print("\n  ✓ No data drift detected")
else:
    print(f"\n  ⚠ ALERT: Drift detected in: {drift_alerts}")

# Now simulate drifted traffic
print("\n--- Simulating drifted traffic (missed_payments +3) ---\n")
drifted_traffic = simulate_requests(200, drift=True,
                                     drift_feature="missed_payments",
                                     drift_amount=3)

for feature in ["missed_payments"]:
    incoming_mean = drifted_traffic[feature].mean()
    baseline_mean = baseline[feature]["mean"]
    baseline_std  = baseline[feature]["std"]
    deviation     = abs(incoming_mean - baseline_mean) / baseline_std
    status        = "OK" if deviation < 2.0 else "⚠ DRIFT DETECTED"
    print(f"  {feature:<25} baseline={baseline_mean:.2f}  "
          f"incoming={incoming_mean:.2f}  "
          f"deviation={deviation:.2f}σ  [{status}]")

# ============================================================
# CHECK 2 — Prediction drift detection
# Is the label distribution shifting from what we expect?
# ============================================================

print("\n" + "="*55)
print("CHECK 2: Prediction Drift Detection")
print("="*55)

preds        = model.predict(normal_traffic[FEATURE_COLS])
pred_labels  = encoder.inverse_transform(preds)
pred_counts  = Counter(pred_labels)
total        = len(pred_labels)

print(f"\n  Expected distribution: {baseline_label_dist}")
print(f"  Observed distribution:")

pred_alerts = []
for label in ["LOW", "MEDIUM", "HIGH"]:
    observed  = pred_counts.get(label, 0) / total
    expected  = baseline_label_dist[label]
    deviation = abs(observed - expected)
    status    = "OK" if deviation < 0.15 else "⚠ DRIFT DETECTED"
    if deviation >= 0.15:
        pred_alerts.append(label)
    print(f"    {label:<8} expected={expected:.2f}  "
          f"observed={observed:.2f}  "
          f"deviation={deviation:.2f}  [{status}]")

if not pred_alerts:
    print("\n  ✓ No prediction drift detected")
else:
    print(f"\n  ⚠ ALERT: Prediction drift in labels: {pred_alerts}")

# ============================================================
# CHECK 3 — Performance drift
# We have 20 labelled samples where we know the true answer
# Check if model is still getting them right
# ============================================================

print("\n" + "="*55)
print("CHECK 3: Performance Drift Detection")
print("="*55)

# These are "ground truth" samples — tenants whose outcomes we know
labelled_samples = pd.DataFrame([
    {"missed_payments":5, "inspection_score":3.0, "complaints_received":3,
     "tenancy_months":2,  "maintenance_calls":6,  "property_size":1, "true_label":"HIGH"},
    {"missed_payments":0, "inspection_score":9.5, "complaints_received":0,
     "tenancy_months":60, "maintenance_calls":1,  "property_size":3, "true_label":"LOW"},
    {"missed_payments":2, "inspection_score":6.0, "complaints_received":1,
     "tenancy_months":24, "maintenance_calls":3,  "property_size":2, "true_label":"MEDIUM"},
    {"missed_payments":4, "inspection_score":4.0, "complaints_received":2,
     "tenancy_months":5,  "maintenance_calls":5,  "property_size":1, "true_label":"HIGH"},
    {"missed_payments":0, "inspection_score":8.5, "complaints_received":0,
     "tenancy_months":48, "maintenance_calls":1,  "property_size":2, "true_label":"LOW"},
])

X_labelled  = labelled_samples[FEATURE_COLS]
true_labels = labelled_samples["true_label"].values
preds       = encoder.inverse_transform(model.predict(X_labelled))

correct  = sum(p == t for p, t in zip(preds, true_labels))
accuracy = correct / len(true_labels)

PERFORMANCE_THRESHOLD = 0.70

print(f"\n  Labelled samples: {len(true_labels)}")
print(f"  Correct:          {correct}")
print(f"  Accuracy:         {accuracy:.2f}")
print(f"  Threshold:        {PERFORMANCE_THRESHOLD}")

for i, (pred, true) in enumerate(zip(preds, true_labels)):
    status = "✓" if pred == true else "✗ WRONG"
    print(f"    Sample {i+1}: predicted={pred:<8} true={true:<8} {status}")

if accuracy >= PERFORMANCE_THRESHOLD:
    print(f"\n  ✓ Performance OK — {accuracy:.0%} on labelled samples")
else:
    print(f"\n  ⚠ ALERT: Performance dropped to {accuracy:.0%} — retrain recommended")

# ============================================================
# SUMMARY REPORT
# ============================================================

print("\n" + "="*55)
print("MONITORING SUMMARY")
print("="*55)

report = {
    "timestamp":        datetime.now().isoformat(),
    "model_version":    metadata["version"],
    "data_drift":       len(drift_alerts) > 0,
    "prediction_drift": len(pred_alerts) > 0,
    "performance_ok":   accuracy >= PERFORMANCE_THRESHOLD,
    "performance":      round(accuracy, 4),
    "alerts":           drift_alerts + pred_alerts
}

print(f"\n  Model version:      {report['model_version']}")
print(f"  Data drift:         {'⚠ YES' if report['data_drift'] else '✓ None'}")
print(f"  Prediction drift:   {'⚠ YES' if report['prediction_drift'] else '✓ None'}")
print(f"  Performance:        {report['performance']:.0%} "
      f"({'✓ OK' if report['performance_ok'] else '⚠ RETRAIN'})")

with open("monitoring_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n  Report saved: monitoring_report.json")
print("\nMonitoring complete.")