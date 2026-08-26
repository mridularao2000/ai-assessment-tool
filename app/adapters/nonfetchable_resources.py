"""Per-resource guidance for grounding exam generation in real, current
content — and for the resources that genuinely cannot be fetched.

Generalized beyond the four named exceptions: any resource label is
substring-matched against NON_FETCHABLE_NOTES; anything that doesn't match
gets the generic search-then-fetch instruction, with the same "fall back to
general knowledge and say so" guidance on any fetch failure — not just the
four named cases. This is what makes the four exceptions a robustness
property rather than a hardcoded four-item switch.
"""
from __future__ import annotations

# Matched by substring against the lowercased resource label.
NON_FETCHABLE_NOTES: dict[str, str] = {
    "system design interview": (
        "a paid book — nothing to fetch. Generate questions from general "
        "knowledge of the topic. Do not reproduce or closely paraphrase "
        "specific passages from the book (copyright)."
    ),
    "bytebytego": (
        "free posts fetch fine, but paid content doesn't exist to you — "
        "don't assume access. If a fetch attempt fails or the content "
        "looks paywalled, fall back to general knowledge for that portion."
    ),
    "chrome devtools memory panel": (
        "a hands-on activity, not a document — there is nothing to fetch. "
        "Generate questions about the underlying debugging skill from "
        "general knowledge, not \"the resource.\""
    ),
    "bigfrontend.dev/react": (
        "interactive/JS-rendered content — a plain fetch may not reliably "
        "pull the actual problem content. Treat as unreliable to fetch; "
        "prefer general knowledge for anything you can't verify was "
        "actually retrieved."
    ),
}


def build_resource_guidance(resources: list[str]) -> str:
    """Build the per-resource instruction block interpolated into the
    generation prompt as {resource_guidance}. Empty string when there are
    no resources (e.g. standalone calls, which never pass this)."""
    if not resources:
        return ""

    lines = [
        "Resource-by-resource guidance — follow this for EVERY resource "
        "listed above before writing questions:",
    ]
    for resource in resources:
        lower = resource.lower()
        note = next(
            (n for key, n in NON_FETCHABLE_NOTES.items() if key in lower), None
        )
        if note:
            lines.append(
                f'- "{resource}": Do NOT attempt to fetch this — {note} '
                "In the generated exam, include a brief notice that this "
                "portion draws on general knowledge rather than the "
                "fetched source, so results aren't silently mislabeled as "
                "resource-grounded when they aren't."
            )
        else:
            lines.append(
                f'- "{resource}": Use web_search to find the current real '
                "page for this resource, then web_fetch it, and ground "
                "related questions in its actual current content (this "
                "matters most for anything that changes over time). If "
                "search or fetch fails for any reason, fall back to "
                "general knowledge and include the same kind of notice — "
                "never fabricate content as if it were read."
            )
    return "\n".join(lines)
