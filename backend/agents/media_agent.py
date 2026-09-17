"""
The media agent: what do the pictures say, and are they even from now?

Stage 5 does the work — the local index lookup, the reverse image
search, the CLIP consistency check, and the DATE_MISMATCH flag that turns
a defensible sentence over a ten-year-old photograph into `misleading`.
This agent decides whether any of that is worth running and which parts
of it can run at all.

Three decisions, and only three:

  * **Is there media?** No images means `skipped`, not a failure and not
    an abstention. A text forward has no pictures to be wrong about.
  * **How many frames?** `images.keyframes` first, on its own, so the
    number of pictures actually worth checking is a recorded decision
    rather than a side effect inside stage 5. A sixty-frame video with
    two distinct shots is two checks.
  * **Which checks?** Reverse image search needs an API key and CLIP
    needs `sentence-transformers`; both probes are the tools' own. Stage 5
    takes `run_reverse_search` and `run_consistency` switches, so the
    agent turns off what cannot run instead of letting it fail inside.

The findings it hands to verification are the ones the verdict rules
already act on: recycled pictures (a date mismatch) and caption
mismatches. Those are read back off the graph rather than counted here,
because the graph is what `backend.verdict` will read too, and two
sources of truth about a recycled photograph is one too many.
"""

from .base import Agent
from .context import MEDIA_TOOLS


class MediaAgent(Agent):
    agent = "media"
    description = "Verify images and video frames: recycled media, caption mismatch."
    tools = ("images.keyframes", "images.collect") + MEDIA_TOOLS

    def run(self, context, step):
        images = context.images()

        if not images:
            return self.skip("this packet has no images or video frames")

        if not context.option("images", True):
            return self.skip("media checks are switched off for this run")

        if context.graph is None:
            return self.fail("there is no graph to write image findings into")

        kept, _call = context.call(
            "images.keyframes", images, _args={"images": len(images)}
        )
        kept = kept if kept is not None else images

        reverse = context.tools.available("images.reverse_search")
        consistency = context.tools.available("images.consistency")

        data = {
            "images": len(images),
            "keyframes": len(kept),
            "dropped_frames": len(images) - len(kept),
            "checks": {
                "local_index": True,          # offline, always available
                "reverse_search": reverse,
                "consistency": consistency,
            },
            "recycled": [],
            "mismatched": [],
            "errors": [],
        }

        report, call = context.call(
            "images.collect",
            context.claimset,
            context.graph,
            context.packet,
            run_reverse_search=reverse,
            run_consistency=consistency,
            _args={
                "keyframes": len(kept),
                "reverse_search": reverse,
                "consistency": consistency,
            },
        )

        if report is None:
            data["errors"].append(call.error or "the image collector returned nothing")

            return self.fail(call.error or "image checks failed", data=data)

        data.update(report.as_dict())
        data["recycled"] = self._recycled(context)
        data["mismatched"] = self._mismatched(context)

        findings = (
            data["index_matches"] + data["reverse_matches"] + data["mismatches"]
        )

        if not findings:
            return self.abstain(
                "the pictures were checked and nothing is known about them: "
                "no archive match, no caption mismatch",
                data=data,
                handoffs=["verification"],
            )

        notes = [
            f"{data['images']} picture(s) checked"
            + (f", {data['dropped_frames']} near-identical frame(s) dropped"
               if data["dropped_frames"] else "")
        ]

        if data["recycled"]:
            notes.append(
                f"{len(data['recycled'])} picture(s) were online before this message"
            )

        if data["mismatches"]:
            notes.append(
                f"{data['mismatches']} caption mismatch(es): the picture does not "
                "show what the claim describes"
            )

        #  A dated picture is hard evidence; a bare archive match is a
        #  lead. The confidence says which of the two this is.
        confidence = 0.85 if data["recycled"] else (
            0.6 if data["mismatches"] else 0.4
        )

        return self.ok(
            data=data,
            confidence=confidence,
            notes=notes,
            handoffs=["verification"],
        )

    def _recycled(self, context):
        """Images the graph has flagged as older than the message."""
        return [
            {
                "image": node_id,
                "path": attrs.get("path"),
                "first_seen": attrs.get("first_seen"),
            }
            for node_id, attrs in context.graph.images()
            if attrs.get("date_mismatch")
        ]

    def _mismatched(self, context):
        """
        Claims a picture was found to contradict.

        Stage 5 writes these with `method="clip"` and stance `refutes`,
        which is exactly what the verdict rules look for, so the agent
        reports the same thing rather than recomputing it.
        """
        found = []

        for claim_id, _attrs in context.graph.claims():
            for item in context.graph.evidence_for(claim_id, stance="refutes"):
                if item.get("method") != "clip":
                    continue

                found.append(
                    {
                        "claim_id": claim_id,
                        "evidence_id": item["id"],
                        "reason": item.get("best_passage") or item.get("snippet"),
                        "similarity": (item.get("meta") or {}).get("similarity"),
                    }
                )

        return found
