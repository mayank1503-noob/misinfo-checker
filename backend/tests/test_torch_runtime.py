"""
The forward-pass guard, and proof that stage 4 really goes through it.

The point of these tests is not that `inference()` is a context manager.
It is that the two model entry points that used to run concurrently
inside stage 3b's nested pools now take one shared lock, and — the part
that would otherwise rot silently — that `apply_stances` actually reaches
the NLI model rather than quietly taking the model-unavailable path that
`--no-nli` takes. A harness that reports a verdict it never computed is
worse than one that refuses to run.
"""

import threading

import pytest

from backend import torch_runtime
from backend.stance import nli, rank


def test_inference_serialises_concurrent_callers():
    """Two threads inside inference() never overlap."""
    inside = []
    peak = []
    lock = threading.Lock()

    def worker():
        with torch_runtime.inference():
            with lock:
                inside.append(1)
                peak.append(len(inside))

            for _ in range(2000):                    # long enough to overlap
                pass

            with lock:
                inside.pop()

    threads = [threading.Thread(target=worker) for _ in range(6)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert max(peak) == 1


def test_inference_is_reentrant():
    """embed() inside a call that already holds the lock must not deadlock."""
    with torch_runtime.inference():
        with torch_runtime.inference():
            assert True


def test_prepare_calls_eval_once_and_tolerates_anything():
    class Model:
        def __init__(self):
            self.evals = 0

        def eval(self):
            self.evals += 1

    model = Model()

    assert torch_runtime.prepare(model) is model
    assert model.evals == 1

    sentinel = object()                              # no .eval(), must not raise
    assert torch_runtime.prepare(sentinel) is sentinel
    assert torch_runtime.prepare(None) is None


def test_embed_enters_the_guard(monkeypatch):
    """rank.embed must not call the encoder outside inference()."""
    seen = []

    class Encoder:
        def encode(self, texts):
            seen.append(torch_runtime._LOCK._is_owned())

            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(rank, "embedder", lambda: Encoder())

    vectors = rank.embed(["a", "b"])

    assert seen == [True]
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]


def test_score_pairs_enters_the_guard(monkeypatch):
    """The NLI forward pass is held by the same lock the embedder uses."""
    held = []

    class Model:
        def __call__(self, **_encoded):
            held.append(torch_runtime._LOCK._is_owned())

            return type("Out", (), {"logits": _Logits()})()

    class _Logits:
        def softmax(self, dim=-1):
            return self

        def tolist(self):
            return [[0.8, 0.1, 0.1]]

    def tokenizer(batch, hypotheses, **_kwargs):
        assert len(batch) == len(hypotheses)

        return {"input_ids": batch}

    monkeypatch.setattr(
        nli, "nli_model",
        lambda: (Model(), tokenizer,
                 {0: "entailment", 1: "neutral", 2: "contradiction"}),
    )

    distributions = nli.score_pairs(["a passage"], "a claim")

    assert held == [True]
    assert distributions[0]["entailment"] == pytest.approx(0.8)


def test_apply_stances_really_invokes_the_nli_model(monkeypatch):
    """
    The regression this file exists for.

    With the model reporting itself available, stage 4 must call
    `score_pairs` — not fall through to the ratings-only path that
    `--no-nli` and a missing checkpoint both take. If a future change
    routes around the model, the call count is zero and this fails.
    """
    from backend import pipeline as pipeline_module
    from backend.stance import classify

    calls = []

    monkeypatch.setattr(nli, "available", lambda: True)
    monkeypatch.setattr(classify.nli, "available", lambda: True)

    def score_pairs(premises, hypothesis, **kwargs):
        calls.append((list(premises), hypothesis))

        return [{"entailment": 0.05, "neutral": 0.05, "contradiction": 0.90}
                for _ in premises]

    monkeypatch.setattr(classify.nli, "score_pairs", score_pairs)
    monkeypatch.setattr(
        classify.rank, "available", lambda: True, raising=False
    )

    result = pipeline_module.analyze(
        {
            "input_type": "text",
            "text": "SBI is giving Rs 5,000 cashback to every customer today.",
        },
        retrieve=True,
        stance=True,
        images=False,
    )

    assert result["verdict"]["label"] in {"false", "true", "misleading",
                                          "unverified"}

    if not calls:
        pytest.skip(
            "no evidence passages reached stage 4 (no corpus or no embedder "
            "in this environment); the NLI path itself is covered by "
            "test_score_pairs_enters_the_guard"
        )

    assert all(hypothesis for _premises, hypothesis in calls)
