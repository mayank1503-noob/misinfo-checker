"""
The evaluation harness: how well does this thing actually work?

Every other stage reports what it *did* — claims extracted, candidates
retrieved, stances scored. None of them says whether any of it was right.
This package does, by running a labelled set of messages through
`backend.pipeline.analyze` and scoring the answers.

    python -m backend.evaluation                  # fast; stages 1-3b
    python -m backend.evaluation --mode full      # + stance and verdict
    python -m backend.evaluation --mode full --compare   # + the agent layer,
                                                          # scored against it

    from backend.evaluation import load, run, summarize
    summary = summarize(run(load(), mode="retrieval"))

Three things about it are deliberate.

**The safety number is separate from the accuracy number.** Missing a
rumour leaves the user where they were; calling a true message false is
the system doing harm. `false_accusations` is reported on its own line,
it is the only thing `--gate` fails on, and it should be zero.

**Cases whose answer is in the demo index are scored apart from cases
whose answer is not.** 24 of the 41 cases can be settled from
`data/seed_factchecks.jsonl`; the other 17 cannot be settled at all, and
for those the correct behaviour is to say nothing. Blending the two would
let a gain in one hide a loss in the other.

**A mode that did not run a stage reports None for it, not zero.** With
stage 4 off, every claim is trivially `unverified`; recording that as a
prediction would invent an accuracy figure out of a stage that never ran.

**The agent layer is evaluated as a change, not as a system.** It runs
the same stages through the same rules, so a second accuracy table would
be the first one again. What `--compare` reports instead is the
difference: which labels moved and in which direction, whether the safety
number moved, what it cost per case, and what the coordination itself did
(routes chosen, agents that abstained, escalations and what they yielded).

See DATA.md for what the labelled set has to contain, and why the numbers
it produces are narrower than they look.
"""

from .dataset import Case, load, select
from .metrics import agent_scores, compare, gate, summarize
from .runner import MODES, run, run_case, run_comparison

__all__ = [
    "Case",
    "load",
    "select",
    "run",
    "run_case",
    "run_comparison",
    "summarize",
    "agent_scores",
    "compare",
    "gate",
    "MODES",
]
