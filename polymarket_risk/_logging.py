"""Structured-style logging on top of stdlib logging.

The parent bot uses structlog. This package keeps the same call shape
(event name plus keyword pairs) without the dependency, so the risk engine
has zero runtime requirements. Event names are preserved verbatim.
"""

from __future__ import annotations

import logging
from typing import Any


class KeyValueLogger:
    """Renders ``log.info("event", key=value)`` as ``event key=value``."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @staticmethod
    def _render(event: str, kwargs: dict[str, Any]) -> str:
        if not kwargs:
            return event
        pairs = " ".join(f"{key}={value}" for key, value in kwargs.items())
        return f"{event} {pairs}"

    def _log(self, level: int, event: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._logger.log(level, self._render(event, kwargs), exc_info=exc_info)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, exc_info=exc_info, **kwargs)

    def error(self, event: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, exc_info=exc_info, **kwargs)

    def critical(self, event: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, exc_info=exc_info, **kwargs)


def get_logger(name: str = "polymarket_risk") -> KeyValueLogger:
    return KeyValueLogger(logging.getLogger(name))
