from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/jwks.json")
def jwks() -> dict:
    """Serve JWKS from ./security/certs/jwks.json if present.

    Rotate keys externally (e.g., CI/cron) every 7 days by updating the file.
    """
    path = Path("./security/certs/jwks.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="JWKS not available")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Invalid JWKS") from exc

