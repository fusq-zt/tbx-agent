"""Security primitives for the API trust boundary."""

from .identity import (
    AuthenticationError,
    ProxyHMACAuthenticator,
    TrustedIdentity,
    build_signed_proxy_headers,
    production_blockers,
)

__all__ = [
    "AuthenticationError",
    "ProxyHMACAuthenticator",
    "TrustedIdentity",
    "build_signed_proxy_headers",
    "production_blockers",
]
