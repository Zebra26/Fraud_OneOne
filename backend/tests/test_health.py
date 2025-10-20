from fastapi.testclient import TestClient

from backend.api.main import app


def test_root_health_endpoint():
    client = TestClient(app)
    response = client.get("/admin/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "dependencies" in body

