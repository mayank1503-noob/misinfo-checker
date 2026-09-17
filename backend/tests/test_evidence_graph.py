"""
The evidence graph moved from `backend.evidence` to `backend.graph`.

Its behaviour is covered by `test_graph_store.py`, which tests the
NetworkX store that now holds it. What is left here is the compatibility
shim: an import written against the old layout must still resolve, and it
must warn rather than silently hand back something different.

See DECISIONS.md (D3) for why the module moved.
"""

import warnings

import pytest


def test_the_old_import_path_still_gives_the_graph():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        from backend.evidence.graph import EvidenceGraph as Shimmed

    from backend.graph import EvidenceGraph

    assert Shimmed is EvidenceGraph


def test_the_old_import_path_warns():
    import importlib

    import backend.evidence.graph as shim

    with pytest.warns(DeprecationWarning, match="moved to backend.graph"):
        importlib.reload(shim)


def test_backend_evidence_is_now_the_retrieval_package():
    import backend.evidence as evidence

    # The package exports retrieval, not a graph.
    assert hasattr(evidence, "collect_evidence")
    assert hasattr(evidence, "EvidenceCandidate")
    assert not hasattr(evidence, "EvidenceGraph")
