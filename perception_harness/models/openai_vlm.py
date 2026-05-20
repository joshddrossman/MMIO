"""OpenAI Chat Completions adapter (vision).

Contract every model adapter must satisfy:

    query(messages: list[dict], *, model_id: str, **kwargs) -> dict | str

The runner accepts either form:

  - str  -> treated as the assistant text, no metadata captured.
  - dict -> must have a "text" key (the assistant text); optional keys
            below are written to per_question.jsonl for post-analysis.

Optional dict keys this adapter populates:
  - "usage": prompt/completion/total token counts (incl. cached + reasoning)
  - "finish_reason": "stop" | "length" | "tool_calls" | "content_filter"
  - "model_resolved": the snapshot the API actually used (e.g. gpt-4o-2024-08-06)
  - "system_fingerprint": OpenAI's backend stack identifier
  - "request_id": chatcmpl-... handle (useful for support / debugging)
  - "n_turns": number of API round-trips (1 for single-shot)
  - "per_turn_finish_reasons": list of finish_reason strings, one per turn
  - "conversation": post-loop msgs trace, with image data stripped
                    (None for single-shot; populated by the tools adapter)

`messages` is the OpenAI-format list returned by a prompt builder. The
adapter doesn't need to know about tasks or scoring — it just shuttles the
messages to the API and returns the raw assistant text + metadata.

Other model adapters (anthropic_vlm, google_vlm, ...) re-shape the
messages internally so the rest of the harness stays provider-agnostic.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


NAME = "openai_vlm"


def _usage_to_dict(usage) -> Optional[Dict[str, Any]]:
    """Convert a ChatCompletion.usage Pydantic model into a plain dict.

    Newer SDKs nest reasoning_tokens inside completion_tokens_details and
    cached_tokens inside prompt_tokens_details — model_dump captures both.
    """
    if usage is None:
        return None
    try:
        return usage.model_dump(exclude_none=True)
    except AttributeError:
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }


def query(messages: List[Dict[str, Any]], *,
          model_id: str = "gpt-5") -> Dict[str, Any]:
    """Call OpenAI Chat Completions and return text + capture metadata."""
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or to "
            "a .env file at the project root.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai>=1.40 required. `pip install --upgrade openai`."
        ) from e

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    choice = resp.choices[0]
    return {
        "text": choice.message.content or "",
        "usage": _usage_to_dict(resp.usage),
        "finish_reason": choice.finish_reason,
        "model_resolved": getattr(resp, "model", None),
        "system_fingerprint": getattr(resp, "system_fingerprint", None),
        "request_id": getattr(resp, "id", None),
        "n_turns": 1,
        "per_turn_finish_reasons": [choice.finish_reason],
        "conversation": None,
    }
