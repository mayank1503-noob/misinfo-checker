"""
Shared plumbing: the disk cache and the fail-soft HTTP helpers that every
external call in the project goes through.
"""

from . import cache, http

__all__ = ["cache", "http"]
