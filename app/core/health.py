from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .config import get_settings


_last_prediction_ts: Optional[float] = None
_model_status: str = "not_loaded"


def set_last_prediction_now() -> None:
    global _last_prediction_ts
    _last_prediction_ts = time.time()


def set_model_status(status: str) -> None:
    global _model_status
    _model_status = status


def get_gpu_status() -> Dict[str, Any]:
    try:
        import torch  # type: ignore

        available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        details = {
            "available": available,
            "device_count": torch.cuda.device_count() if available else 0,
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": getattr(torch.cuda.get_device_properties(i), "major", None),
                }
                for i in range(torch.cuda.device_count())
            ]
            if available
            else [],
        }
        status = "available" if available else "unavailable"
        return {"status": status, "details": details}
    except Exception:
        return {"status": "unavailable", "details": {"available": False}}


def health_payload() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "application": {
            "name": settings.app.name,
            "version": settings.app.version,
            "status": "healthy",
        },
        "model": {
            "loading_status": _model_status,
            "last_successful_prediction_ts": _last_prediction_ts,
        },
        "gpu": get_gpu_status(),
    }
