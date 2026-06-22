"""
promote.py — promote any model version to production

Usage:
    python promote.py                        # promote latest
    python promote.py v20240622_143000       # promote specific version
"""

import json
import joblib
import shutil
import sys
import os

REGISTRY_PATH = "model_registry/registry.json"

def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)

def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def list_versions(registry):
    print("\nAll trained model versions:")
    print("-" * 60)
    for m in registry["models"]:
        marker = " ← PRODUCTION" if m["version"] == registry.get("production") else ""
        print(f"  {m['version']}  acc={m['accuracy']}  f1={m['f1_score']}{marker}")
    print()

def promote(version=None):
    registry = load_registry()
    list_versions(registry)

    if version is None:
        # Default: promote the latest candidate
        candidates = [m for m in registry["models"]]
        if not candidates:
            print("No models found.")
            return
        version = candidates[-1]["version"]
        print(f"No version specified — promoting latest: {version}")

    model_dir = f"model_registry/{version}"
    if not os.path.exists(model_dir):
        print(f"Version {version} not found in registry.")
        return

    # Copy to active model folder
    os.makedirs("model", exist_ok=True)
    shutil.copy(f"{model_dir}/risk_model.pkl",    "model/risk_model.pkl")
    shutil.copy(f"{model_dir}/label_encoder.pkl", "model/label_encoder.pkl")
    shutil.copy(f"{model_dir}/metadata.json",     "model/metadata.json")

    # Update registry
    for m in registry["models"]:
        m["status"] = "candidate"
        if m["version"] == version:
            m["status"] = "production"

    registry["production"] = version
    save_registry(registry)

    print(f"Promoted {version} to production.")
    print(f"Active model updated at model/")

if __name__ == "__main__":
    version_arg = sys.argv[1] if len(sys.argv) > 1 else None
    promote(version_arg)