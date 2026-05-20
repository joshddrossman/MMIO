"""Zero-shot prompt builder.

Returns OpenAI-style messages with the rendered map as a base64 data URL.
The task module supplies the actual question text — this module is just
responsible for assembling the system + user messages and embedding the
image.

Contract every prompt module must satisfy:

    build(task_name: str, question: str, image_b64: str) -> list[dict]

The returned list is suitable for OpenAI Chat Completions `messages`. The
OpenAI VLM adapter passes it through verbatim; the Anthropic adapter
translates the schema in its own module.
"""
from __future__ import annotations

from typing import Any, Dict, List


NAME = "zero_shot"

# Prompt capability — read by eval_perception.py to build view_info.
# True ⇒ the prompt embeds the rendered map in the user message.
USES_IMAGE = True


SYSTEM_TEXT = (
    "You are a careful visual analyst. You are looking at a map of a "
    "polling-place plan and answering a specific question about what is "
    "visible on the map. Follow the response format the user requests "
    "exactly — no preamble, no markdown fences unless explicitly asked."
)


def build(*, task_name: str, question: str,
          image_b64: str,
          view_info: Dict[str, Any] | None = None,
          **_kwargs) -> List[Dict[str, Any]]:
    """Build a zero-shot multimodal message list.

    `view_info` is accepted for runner-contract compatibility but
    intentionally unused here — the zero-shot prompt is task-agnostic
    by design. Sibling prompts (with_attribution, etc.) consult
    view_info to pick text variants.
    """
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ],
        },
    ]
