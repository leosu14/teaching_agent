"""URL canonicalisation and stable source ids, so the same document is recognised however it is linked."""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"})
DEFAULT_PORTS = {"http": 80, "https": 443}


def canonical_url(url: str) -> str:
    """Equivalent URLs map to one form: case-insensitive scheme and host, no `www.`, no default port,
    no fragment, no tracking parameters, sorted query, no trailing slash, and http treated as https."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:  # kb://, doi:, file:// ... keep as is apart from case and fragment
        return urlunsplit((scheme, parts.netloc.lower(), parts.path, parts.query, ""))
    host = (parts.hostname or "").lower().removeprefix("www.")
    port = parts.port
    netloc = host if port in (None, DEFAULT_PORTS.get(scheme)) else f"{host}:{port}"
    query = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    )
    path = parts.path.rstrip("/") or ""
    return urlunsplit(("https", netloc, path, urlencode(query), ""))


def source_id_for(url: str) -> str:
    """Stable id derived from the canonical URL: the same document always gets the same id."""
    return "src_" + hashlib.sha256(canonical_url(url).encode()).hexdigest()[:12]


def host_matches(url: str, domain: str) -> bool:
    """Whether the URL's host is `domain` or one of its subdomains (case-insensitive, `www.` ignored)."""
    host = (urlsplit(url.strip()).hostname or "").lower().removeprefix("www.")
    domain = domain.strip().lower().removeprefix("www.").rstrip(".")
    return bool(domain) and (host == domain or host.endswith("." + domain))
