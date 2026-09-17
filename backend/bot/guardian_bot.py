"""
The bot front door: a forwarded message in, a reply out.

`process_message` used to classify a message and echo it back. It now
runs the pipeline and returns something worth sending: a verdict, a
sentence explaining it, and the sources behind it.

The reply is written for the person who forwarded the message, which
changes what belongs in it:

  * **Lead with what to do.** "Don't forward this" is the useful part;
    the evidence is the justification, not the headline.
  * **Name the source.** "PIB rated this false" persuades where "our
    system says false" does not.
  * **Say when we don't know.** `unverified` is the honest answer for
    most forwards, and a reply that hedges keeps its credibility for the
    cases where it does not have to.
  * **Say when it is demo data.** A verdict resting on the demo index
    says so, out loud, in the reply.

Nothing here raises: a bot that crashes on a malformed forward stops
answering everyone.
"""

import logging

from .filters import classify_message


log = logging.getLogger(__name__)


# What to tell someone for each verdict, in the order of what they should
# do about it.
ADVICE = {
    "false": "Don't forward this - it isn't true.",
    "misleading": "Be careful with this one - it's misleading.",
    "true": "This one checks out.",
    "disputed": "Sources disagree about this - don't treat it as settled.",
    "unverified": "I couldn't verify this. Treat it with caution.",
}

EMOJI = {
    "false": "❌",
    "misleading": "⚠️",
    "true": "✅",
    "disputed": "🤔",
    "unverified": "❓",
}

MAX_REASONS = 2
MAX_REPLY_CHARS = 900


def format_reply(result):
    """Turn a pipeline result into the text a bot sends back."""
    verdict = result.get("verdict") or {}
    label = verdict.get("label", "unverified")

    lines = [f"{EMOJI.get(label, '❓')} {ADVICE.get(label, ADVICE['unverified'])}"]

    claims = verdict.get("claims") or []

    if not claims:
        lines.append("I couldn't find a factual claim in this message to check.")

        return "\n".join(lines)

    worst = claims[0]
    lines.append("")
    lines.append(f"Claim: {(worst.get('claim') or '')[:200]}")
    lines.append(worst.get("explanation", ""))

    reasons = worst.get("reasons") or []

    if reasons:
        lines.append("")
        lines.append("Why:")

        for reason in reasons[:MAX_REASONS]:
            lines.append(f"  - {reason}")

    if len(claims) > 1:
        others = len(claims) - 1
        lines.append("")
        lines.append(f"({others} other claim{'s' if others > 1 else ''} checked.)")

    if worst.get("demo_only"):
        lines.append("")
        lines.append(
            "Note: this is running on the demo evidence index, not on live "
            "fact-checks."
        )

    reply = "\n".join(line for line in lines if line is not None)

    if len(reply) > MAX_REPLY_CHARS:
        reply = reply[:MAX_REPLY_CHARS].rsplit(" ", 1)[0] + " ..."

    return reply


def process_message(text, analyze=None, **kwargs):
    """
    Check one incoming message.

    `analyze` is injectable so a caller can pass a cheaper configuration
    (or the tests can pass a stub) without this module knowing about it.
    Returns the verdict, the reply text, and the full result for anything
    that wants to dig further.
    """
    message_type = classify_message(text)

    if message_type == "empty":
        return {
            "status": "ignored",
            "reason": "Empty message",
            "reply": "Send me a message, a link, or a screenshot and I'll check it.",
        }

    if analyze is None:
        from ..pipeline import analyze_link, analyze_text

        analyze = analyze_link if message_type == "link" else analyze_text

    body = text.strip()

    if message_type == "link":
        body = _first_url(body) or body

    try:
        result = analyze(body, **kwargs)
    except Exception as error:                       # fail soft, never raise
        log.warning("analysis failed (%s): %s", type(error).__name__, error)

        return {
            "status": "error",
            "input_type": message_type,
            "text": text.strip(),
            "reason": f"{type(error).__name__}: {error}",
            "reply": (
                "Something went wrong while checking that. Please try again."
            ),
        }

    verdict = result.get("verdict") or {}

    return {
        "status": "checked",
        "input_type": message_type,
        "text": text.strip(),
        "label": verdict.get("label", "unverified"),
        "confidence": verdict.get("confidence", 0.0),
        "reply": format_reply(result),
        "result": result,
    }


def _first_url(text):
    """The first URL in a message, normalised to something fetchable."""
    for token in (text or "").split():
        cleaned = token.strip().strip(".,;:!?()[]<>\"'")

        if cleaned.startswith(("http://", "https://")):
            return cleaned

        if cleaned.startswith("www."):
            # filters.contains_url accepts bare `www.`, but the link
            # analyzer needs a scheme to fetch anything.
            return "https://" + cleaned

    return None


if __name__ == "__main__":
    message = "SBI is giving Rs 5,000 cashback, forward this to 10 people!"

    print(process_message(message)["reply"])
