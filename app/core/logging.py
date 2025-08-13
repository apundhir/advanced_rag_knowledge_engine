from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from logging import LogRecord
from pathlib import Path
from typing import Any, Optional

from pythonjsonlogger import jsonlogger

from .config import get_settings

# Context variable for trace ID that can be used across the request lifecycle
trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)


class JsonFormatter(jsonlogger.JsonFormatter):
    """JSON log formatter with a fixed schema and optional trace_id support."""

    def add_fields(self, log_record: dict[str, Any], record: LogRecord, message_dict: dict[str, Any]) -> None:  # type: ignore[override]
        super().add_fields(log_record, record, message_dict)
        log_record.setdefault("timestamp", self.formatTime(record, self.datefmt))
        log_record.setdefault("level", record.levelname)
        log_record.setdefault("logger", record.name)
        log_record.setdefault("message", record.getMessage())
        log_record.setdefault("file", record.filename)
        log_record.setdefault("line", record.lineno)
        log_record.setdefault("function", record.funcName)
        # Inject trace id from context if not already present
        trace_id: Optional[str] = getattr(record, "trace_id", None) or trace_id_ctx.get()
        log_record.setdefault("trace_id", trace_id)


class TraceIdFilter(logging.Filter):
    """Ensure every record has a trace_id attribute so formatter can include it."""

    def filter(self, record: LogRecord) -> bool:  # noqa: A003 - intentionally named filter
        if not getattr(record, "trace_id", None):
            record.trace_id = trace_id_ctx.get()
        return True


def configure_logging() -> None:
    """Configure root logging with JSON formatter based on app settings."""

    settings = get_settings()
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    # Ensure log directory exists if writing to file
    handlers: list[logging.Handler] = []
    if settings.logging.log_file:
        log_path = Path(settings.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        handlers.append(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    handlers.append(stream_handler)

    for handler in handlers:
        if settings.logging.json:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        handler.addFilter(TraceIdFilter())

    logging.basicConfig(level=level, handlers=handlers, force=True)


def new_trace_id() -> str:
    """Generate a new trace ID and set it in context."""

    trace = uuid.uuid4().hex
    trace_id_ctx.set(trace)
    return trace
