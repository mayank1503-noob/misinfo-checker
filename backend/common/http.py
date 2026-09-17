"""
The one place this project talks to the internet.

Two functions — `get_json` and `post_json` — wrap `requests` with the
three properties every retriever needs and none of them should implement
twice:

  * **Fail soft.** A timeout, a 429, a DNS failure, malformed JSON or a
    missing API key all return `None` after a log line. Nothing raises.
    One dead retriever must cost the pipeline that retriever's evidence,
    not the verdict.
  * **Cached.** Successful responses go to `backend/common/cache.py`,
    keyed by the full request. Re-running the pipeline on the same
    message costs nothing and works offline.
  * **Mockable.** Because every retriever goes through these two
    functions, the entire test suite stubs the network at one seam and
    never opens a socket.

Keys are read from the environment at call time, not at import, so a
`.env` loaded late still works and tests can set and unset them freely.
"""

import logging
import os

from . import cache


log = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 10
USER_AGENT = "misinfo-checker/1.0 (+https://github.com/; research prototype)"


def api_key(name):
    """An API key from the environment, or None when it is unset or blank."""
    value = (os.getenv(name) or "").strip()

    return value or None


def _request(method, url, params=None, json_body=None, headers=None,
             timeout=DEFAULT_TIMEOUT, namespace=None, cache_parts=None):
    if namespace:
        parts = cache_parts or [method, url, params, json_body]
        return cache.memoize(
            namespace,
            parts,
            lambda: _request(
                method, url,
                params=params, json_body=json_body, headers=headers,
                timeout=timeout, namespace=None,
            ),
        )

    try:
        import requests
    except ImportError:                              # fail soft, never raise
        log.warning("requests is not installed; %s %s skipped", method, url)
        return None

    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers or {})

    try:
        response = requests.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=request_headers,
            timeout=timeout,
        )
    except Exception as error:                       # fail soft, never raise
        log.warning("%s %s failed (%s): %s", method, url, type(error).__name__, error)
        return None

    if response.status_code >= 400:
        log.warning(
            "%s %s returned %s: %s",
            method, url, response.status_code, response.text[:200],
        )
        return None

    try:
        return response.json()
    except ValueError:
        log.warning("%s %s returned non-JSON content", method, url)
        return None


def get_json(url, params=None, headers=None, timeout=DEFAULT_TIMEOUT,
             namespace=None, cache_parts=None):
    """GET a JSON document, or None on any failure."""
    return _request(
        "GET", url,
        params=params, headers=headers, timeout=timeout,
        namespace=namespace, cache_parts=cache_parts,
    )


def post_json(url, json_body=None, headers=None, timeout=DEFAULT_TIMEOUT,
              namespace=None, cache_parts=None):
    """POST a JSON body and read a JSON response, or None on any failure."""
    return _request(
        "POST", url,
        json_body=json_body, headers=headers, timeout=timeout,
        namespace=namespace, cache_parts=cache_parts,
    )


def get_text(url, timeout=DEFAULT_TIMEOUT, namespace=None):
    """
    Fetch a page as text, or None.

    Used by the article fetcher; kept here so page downloads are cached
    and fail soft on the same terms as API calls.
    """
    def download():
        try:
            import requests
        except ImportError:                          # fail soft, never raise
            log.warning("requests is not installed; GET %s skipped", url)
            return None

        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
                timeout=timeout,
            )
        except Exception as error:                   # fail soft, never raise
            log.warning("GET %s failed (%s): %s", url, type(error).__name__, error)
            return None

        if response.status_code >= 400:
            log.warning("GET %s returned %s", url, response.status_code)
            return None

        return response.text

    if namespace:
        return cache.memoize(namespace, ["GET", url], download)

    return download()
