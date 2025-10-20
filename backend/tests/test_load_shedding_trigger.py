import base64
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend.middleware.load_shed import LoadShedMiddleware


def test_load_shedding_trigger(monkeypatch):
    app = FastAPI()
    app.add_middleware(LoadShedMiddleware)

    # Dummy scoring route without auth middleware
    @app.post("/predictions/score")
    def score():
        return JSONResponse({"ok": True})

    # Force CPU percent function to return high
    import backend.middleware.load_shed as ls

    monkeypatch.setattr(ls, "_cpu_percent", lambda: 99.0)

    client = TestClient(app)
    r = client.post("/predictions/score")
    assert r.status_code == 503
    body = r.json()
    assert body.get("status") == "degraded_mode"

