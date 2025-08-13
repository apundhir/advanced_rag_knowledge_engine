from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel
from dotenv import load_dotenv


class AppConfig(BaseModel):
    """Application runtime configuration.

    Attributes:
        name: Application name used in logs and health output.
        version: Semantic version string of the application.
        host: Bind address for the API server.
        port: Internal application port (Docker uses 5000 by default).
        workers: Number of server workers (not used by uvicorn's --reload).
    """

    name: str = "advanced-rag-engine"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 5000
    workers: int = 1


class LoggingConfig(BaseModel):
    """Logging configuration values.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json: Whether to emit JSON logs.
        log_file: Optional file path for log output. If not set, logs go to stdout.
    """

    level: str = "INFO"
    json: bool = True
    log_file: Optional[str] = "logs/app.log"


class DBConfig(BaseModel):
    """Database configuration for local vector store."""

    chroma_path: str = "data/chroma"


class RuntimeConfig(BaseModel):
    """Runtime configuration such as device selection."""

    device: str = "cpu"
    cuda_visible_devices: str = ""


class Settings(BaseModel):
    """Top-level settings object composed from YAML and environment variables."""

    app: AppConfig = AppConfig()
    logging: LoggingConfig = LoggingConfig()
    db: DBConfig = DBConfig()
    runtime: RuntimeConfig = RuntimeConfig()


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _deep_update(target: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    # Flat env overrides commonly used
    env_overrides: dict[str, Any] = {}
    if os.getenv("HOST"):
        env_overrides.setdefault("app", {})["host"] = os.getenv("HOST")
    if os.getenv("PORT"):
        env_overrides.setdefault("app", {})["port"] = int(os.getenv("PORT", "5000"))
    if os.getenv("LOG_LEVEL"):
        env_overrides.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        env_overrides.setdefault("runtime", {})["cuda_visible_devices"] = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if os.getenv("DEVICE"):
        env_overrides.setdefault("runtime", {})["device"] = os.getenv("DEVICE")
    if env_overrides:
        config = _deep_update(config, env_overrides)
    return config


@lru_cache(maxsize=1)
def get_settings(config_path: Optional[str] = None) -> Settings:
    """Load settings from YAML and environment once and cache the result.

    Args:
        config_path: Optional path to YAML config. Defaults to `configs/config.yaml` in repo.

    Returns:
        Settings: Pydantic settings object.
    """

    # Load .env first for environment variable overrides
    load_dotenv(override=False)

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    data = _load_yaml_config(path)
    data = _apply_env_overrides(data)
    return Settings.model_validate(data)
