"""Typed provider errors. Every adapter maps its vendor's failures onto these, so nothing above the provider layer
ever sees a vendor exception.

`transient` says whether the same request may succeed later. It drives fallback (only transient failures move to a
configured fallback provider) and the callers' own retry policies. Provider-level retry is narrower: only the typed
transient failures below (timeout, rate limit, unavailable) are retried by the provider layer.

`possibly_billed` says what the failure tells about the vendor's side, for the spend ledger: True when the vendor
may have processed (and billed) the request (the connection broke after the response started, or a gateway or
server error that can follow execution), False when it certainly did not (the connection was never established), None
when the error type decides (see `failure_state` in invoker.py).
"""

from __future__ import annotations


class ProviderError(Exception):
    """A provider call failed. Base of every provider failure."""

    default_transient = True

    def __init__(self, message: str, *, transient: bool | None = None, provider: str | None = None,
                 status: int | None = None, request_id: str | None = None,
                 possibly_billed: bool | None = None) -> None:
        super().__init__(message)
        self.transient = self.default_transient if transient is None else transient
        self.provider = provider
        self.status = status
        self.request_id = request_id
        self.possibly_billed = possibly_billed


class ProviderTimeout(ProviderError, TimeoutError):
    """The provider did not answer within the configured timeout."""


class ProviderRateLimit(ProviderError):
    """The provider rejected the call for exceeding a rate or quota limit (HTTP 429)."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ProviderUnavailable(ProviderError):
    """Connection failure, overload or a transient 5xx."""


class ProviderAuthenticationError(ProviderError):
    """Missing, invalid or unauthorised credentials (HTTP 401/403). Never retried."""

    default_transient = False


class ProviderInvalidRequest(ProviderError):
    """The request is malformed or not supported by this provider (permanent 4xx, bad parameters). Never retried."""

    default_transient = False


class ProviderResponseError(ProviderError):
    """The provider answered, but with something unusable: an unparseable body, a refusal, a missing field."""

    default_transient = False


class ProviderOfflineError(ProviderError):
    """A network provider was used while offline mode (TEACHING_AGENT_MODE=offline, the default) is on."""

    default_transient = False


RETRYABLE = (ProviderTimeout, ProviderRateLimit, ProviderUnavailable)


def is_retryable(exc: BaseException) -> bool:
    """Provider-level retry: timeouts, rate limits and transient unavailability, and only when still transient."""
    return isinstance(exc, RETRYABLE) and exc.transient
