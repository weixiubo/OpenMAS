"""Structured error taxonomy and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass


class OpenAGORAError(RuntimeError):
    """Base error for OpenAGORA runtime failures."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool,
        code: str,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.code = code


class ConfigurationError(OpenAGORAError):
    """Configuration or bootstrap failures."""

    def __init__(self, message: str, *, code: str = "config_error") -> None:
        super().__init__(message, category="configuration", retryable=False, code=code)


class ExternalDependencyError(OpenAGORAError):
    """Errors from provider/tool/storage dependencies."""

    def __init__(self, message: str, *, retryable: bool, code: str) -> None:
        super().__init__(message, category="external_dependency", retryable=retryable, code=code)


class RuntimeExecutionError(OpenAGORAError):
    """Unexpected runtime errors in internal workflow."""

    def __init__(self, message: str, *, code: str = "runtime_error") -> None:
        super().__init__(message, category="runtime_execution", retryable=False, code=code)


@dataclass(frozen=True)
class ErrorInfo:
    """Serializable classification payload."""

    category: str
    code: str
    retryable: bool
    error_type: str
    message: str


def normalize_error(exc: Exception) -> ErrorInfo:
    """Normalize raw exception into stable structured fields."""
    if isinstance(exc, OpenAGORAError):
        return ErrorInfo(
            category=exc.category,
            code=exc.code,
            retryable=exc.retryable,
            error_type=exc.__class__.__name__,
            message=str(exc),
        )
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        return ErrorInfo(
            category="configuration",
            code="path_error",
            retryable=False,
            error_type=exc.__class__.__name__,
            message=str(exc),
        )
    if isinstance(exc, ValueError):
        return ErrorInfo(
            category="configuration",
            code="validation_error",
            retryable=False,
            error_type=exc.__class__.__name__,
            message=str(exc),
        )
    return ErrorInfo(
        category="runtime_execution",
        code="unhandled_error",
        retryable=False,
        error_type=exc.__class__.__name__,
        message=str(exc),
    )
