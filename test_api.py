import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """API should return healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "version" in data
    print("  PASS: Health check")


def test_predict_high_risk(client):
    """Tenant with many missed payments should be HIGH risk."""
    response = client.post(
        "/predict",
        json={
            "missed_payments":     5,
            "inspection_score":    3.0,
            "complaints_received": 3,
            "tenancy_months":      2,
            "maintenance_calls":   6,
            "property_size":       1
        }
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["risk_level"] == "HIGH"
    assert data["confidence"] > 0.5
    print(f"  PASS: HIGH risk prediction — confidence {data['confidence']}")


def test_predict_low_risk(client):
    """Tenant with clean record should be LOW risk."""
    response = client.post(
        "/predict",
        json={
            "missed_payments":     0,
            "inspection_score":    9.5,
            "complaints_received": 0,
            "tenancy_months":      60,
            "maintenance_calls":   1,
            "property_size":       3
        }
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["risk_level"] == "LOW"
    print(f"  PASS: LOW risk prediction — confidence {data['confidence']}")


def test_missing_fields(client):
    """API should return 400 when fields are missing."""
    response = client.post(
        "/predict",
        json={"missed_payments": 3}
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    print("  PASS: Missing fields handled correctly")


def test_model_info(client):
    """Model info endpoint should return metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert "version" in data
    assert "accuracy" in data
    assert "features" in data
    print(f"  PASS: Model info — version {data['version']}")


if __name__ == "__main__":
    print("\nRunning DataCompany API tests...")
    pytest.main([__file__, "-v"])