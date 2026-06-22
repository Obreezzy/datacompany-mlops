from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import os

app = Flask(__name__)

# Load model and encoder at startup
print("Loading DataCompany risk model...")
model   = joblib.load("model/risk_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")

with open("model/metadata.json") as f:
    metadata = json.load(f)

print(f"Model version {metadata['version']} loaded.")
print(f"Accuracy: {metadata['accuracy']}")

FEATURE_COLS = metadata["features"]


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — used by Kubernetes liveness probe."""
    return jsonify({
        "status":  "healthy",
        "version": metadata["version"],
        "model":   metadata["model_type"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts tenant data and returns risk prediction.

    Request body:
    {
        "missed_payments": 3,
        "inspection_score": 5.5,
        "complaints_received": 1,
        "tenancy_months": 12,
        "maintenance_calls": 2,
        "property_size": 2
    }
    """
    try:
        data = request.json

        # Validate all required fields are present
        missing = [f for f in FEATURE_COLS if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}"
            }), 400

        # Build dataframe
        df   = pd.DataFrame([{col: data[col] for col in FEATURE_COLS}])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0]

        risk_label  = encoder.inverse_transform([pred])[0]
        confidence  = round(float(prob.max()), 3)

        # Build probability breakdown for all classes
        class_probs = {
            cls: round(float(p), 3)
            for cls, p in zip(encoder.classes_, prob)
        }

        return jsonify({
            "risk_level":        risk_label,
            "confidence":        confidence,
            "class_probabilities": class_probs,
            "model_version":     metadata["version"],
            "features_used":     FEATURE_COLS
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """Returns model metadata — useful for auditing."""
    return jsonify(metadata)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)