"""
One place that decides *how* torch is entered, shared by every stage.

Three models live in this process at once — the sentence-transformer the
seed index embeds with, the mDeBERTa the claim stage and stage 4 share,
and (with images on) CLIP. Loading them was already serialised: both
thread pools in `backend/evidence/collector.py` call `warm_up()` on the
calling thread first, because materialising weights inside a worker has
aborted the interpreter on Windows.

Running them was not. Stage 3b searches claims in a pool, and each claim
fans out to a second pool over the retrievers, so `seed_index.search` —
and the embedder forward pass inside it — can be executing on several
worker threads at once while stage 4's entailment weights are resident.
Each concurrent forward allocates its own activation buffers and torch's
own intra-op pool multiplies the thread count, so peak RSS scales with
the pool width rather than with the batch. That is what makes
`--mode full` survive a single case and fall over across several: nothing
is leaked, the peak is simply reached.

So: `inference()` is the only way this codebase enters a torch forward
pass. It takes a process-wide lock and disables autograd, which turns
concurrent model execution back into sequential model execution while
leaving the IO-bound part of retrieval — the HTTP that the pools exist
for — genuinely parallel. `prepare()` puts a freshly loaded module in
eval mode once.

Nothing here changes a single number any stage produces. Serialising a
forward pass and running it without autograd gives bit-identical logits;
it only changes how much memory is live at the peak.
"""

import logging
import os
import threading
from contextlib import contextmanager


log = logging.getLogger(__name__)


# Held for the duration of every forward pass in this process. Reentrant
# because a ranked stance pass embeds inside a call that may already hold
# it, and a plain Lock would deadlock on the second acquire.
_LOCK = threading.RLock()

_THREADS_CONFIGURED = False


def configure_threads():
    """
    Cap torch's intra-op threads when MISINFO_TORCH_THREADS is set.

    Off by default: with the lock below there is no oversubscription to
    fix, and silently changing the thread count would change how every
    other machine performs. It exists because a low-RAM box can still
    want the per-op arenas smaller, and that is a deployment decision,
    not one this module should make for everybody.
    """
    global _THREADS_CONFIGURED

    if _THREADS_CONFIGURED:
        return False

    requested = os.getenv("MISINFO_TORCH_THREADS")

    if not requested:
        return False

    try:
        import torch

        torch.set_num_threads(max(1, int(requested)))
        _THREADS_CONFIGURED = True

        return True
    except Exception as error:                       # fail soft, never raise
        log.warning(
            "could not set torch threads to %r (%s): %s",
            requested, type(error).__name__, error,
        )

        return False


def prepare(model):
    """
    Put a just-loaded module in eval mode and return it.

    Idempotent and forgiving: anything without `.eval()` (a pipeline
    wrapper, a stub in the tests) comes back untouched.
    """
    evaluate = getattr(model, "eval", None)

    if callable(evaluate):
        try:
            evaluate()
        except Exception as error:                   # fail soft, never raise
            log.debug("eval() on %r failed: %s", type(model).__name__, error)

    return model


@contextmanager
def inference():
    """
    Enter a forward pass: one at a time, process-wide, without autograd.

    Safe to use when torch is not installed — it then only takes the
    lock, so the stubbed models in the tests take the same path the real
    ones do.
    """
    configure_threads()

    with _LOCK:
        try:
            import torch
        except ImportError:
            yield
            return

        with torch.inference_mode():
            yield
