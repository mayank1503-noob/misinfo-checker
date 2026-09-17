"""
Shared test fixtures — chiefly, the guarantee that the suite is offline.

Every external call in this project goes through `backend.common.http`,
so blocking those three functions blocks the network for the whole suite,
whatever a test forgets to stub. They return `None`, which is what a
failed call returns anyway, so the fail-soft paths are what gets
exercised rather than an artificial exception.

A test that needs a particular payload monkeypatches the same names in
its own body; monkeypatch applies after this fixture, so the test's stub
wins.

This is autouse and has no opt-out on purpose. A test suite that can
reach the internet is a test suite that passes on a plane and fails in
CI, or quietly spends someone's API quota.
"""

import pytest

from backend.common import http as _http


# Captured at import, before any fixture patches them.
_REAL = {
    "get_json": _http.get_json,
    "post_json": _http.post_json,
    "get_text": _http.get_text,
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        return None

    for name in ("get_json", "post_json", "get_text"):
        monkeypatch.setattr(f"backend.common.http.{name}", blocked)

    yield


@pytest.fixture
def real_http():
    """
    The unpatched `get_json` / `post_json` / `get_text`.

    For the handful of tests that are *about* the HTTP helpers and their
    cache. They still reach no further than a stubbed `requests` module -
    the seam moves down one level, it does not open.
    """
    from backend.common import http

    return _REAL
