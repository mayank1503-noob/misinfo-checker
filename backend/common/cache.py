"""
A disk cache for everything this project fetches from the outside world.

Every external call — fact-check search, web search, article download,
reverse image lookup — goes through here first. That buys three things
the pipeline depends on:

  * a demo that keeps working when the venue's wifi does, or when a free
    API tier runs out mid-presentation;
  * a test suite that never opens a socket;
  * repeatability — the same input produces the same evidence, so a
    verdict can be re-derived and argued with.

Entries are JSON files under `cache/<namespace>/<sha1>.json`, named by a
hash of the request, and they do not expire by default: a fact-check
published in 2019 says the same thing today, and a stale hit is a far
smaller problem than a rate-limited one. Set `CACHE_TTL_DAYS` if you
disagree, or `CACHE_DISABLED=1` to bypass it entirely.

Nothing here raises. A corrupt entry, an unwritable directory or a
read-only filesystem all degrade to "no cache", because a caching layer
that can take the pipeline down is worse than no caching layer.
"""

import hashlib
import json
import logging
import os
import time


log = logging.getLogger(__name__)


DEFAULT_DIR = "cache"


def cache_dir():
    return os.getenv("CACHE_DIR", DEFAULT_DIR)


def disabled():
    return os.getenv("CACHE_DISABLED", "").strip().lower() in ("1", "true", "yes")


def ttl_seconds():
    """0 means "never expires", which is the default."""
    try:
        days = float(os.getenv("CACHE_TTL_DAYS", "0"))
    except ValueError:
        return 0.0

    return max(0.0, days) * 86400


def key_for(*parts):
    """A stable cache key for any combination of JSON-able request parts."""
    blob = json.dumps(
        [part if isinstance(part, (str, int, float, bool, type(None))) else
         json.loads(json.dumps(part, sort_keys=True, default=str))
         for part in parts],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def path_for(namespace, key):
    return os.path.join(cache_dir(), namespace, f"{key}.json")


def get(namespace, key):
    """The cached value, or None on a miss, an expiry, or any failure."""
    if disabled():
        return None

    path = path_for(namespace, key)

    try:
        if not os.path.exists(path):
            return None

        ttl = ttl_seconds()

        if ttl and (time.time() - os.path.getmtime(path)) > ttl:
            return None

        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)["value"]
    except Exception as error:                       # fail soft, never raise
        log.debug("cache read failed for %s/%s: %s", namespace, key, error)
        return None


def put(namespace, key, value):
    """Store a value; returns True when it actually landed on disk."""
    if disabled():
        return False

    path = path_for(namespace, key)

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {"namespace": namespace, "key": key, "stored_at": time.time(),
                   "value": value}

        # Write then rename, so a crash mid-write cannot leave a
        # half-written file that later reads as a corrupt cache hit.
        temporary = f"{path}.tmp"

        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

        os.replace(temporary, path)

        return True
    except Exception as error:                       # fail soft, never raise
        log.debug("cache write failed for %s/%s: %s", namespace, key, error)
        return False


def memoize(namespace, parts, producer):
    """
    Return the cached value for `parts`, else call `producer()` and cache
    what it returns.

    `None` is never cached: it is what every failing call in this project
    returns, and caching a failure would turn one flaky request into a
    permanently empty result.
    """
    key = key_for(*parts)
    hit = get(namespace, key)

    if hit is not None:
        log.debug("cache hit %s/%s", namespace, key)
        return hit

    value = producer()

    if value is not None:
        put(namespace, key, value)

    return value


def clear(namespace=None):
    """Delete one namespace, or the whole cache. Returns files removed."""
    import shutil

    target = os.path.join(cache_dir(), namespace) if namespace else cache_dir()

    if not os.path.isdir(target):
        return 0

    removed = sum(len(files) for _root, _dirs, files in os.walk(target))

    try:
        shutil.rmtree(target)
    except Exception as error:                       # fail soft, never raise
        log.warning("could not clear cache at %s: %s", target, error)
        return 0

    return removed
