"""Typed local inference failure."""


class EgorLocalProviderError(RuntimeError):
    """Normalized failure raised by the frozen EGOR inference adapter."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
