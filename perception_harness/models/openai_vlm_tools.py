"""OpenAI Chat Completions adapter with tool-call loop.

Sibling of openai_vlm.py for the tool-using path. Kept separate so the
single-shot adapter stays simple.

Contract:

    SUPPORTS_TOOLS = True       # the runner uses this to detect the loop

    query(messages, *, model_id, tools=None, tool_dispatcher=None,
          dispatcher_kwargs=None, max_iters=10) -> dict

If `tools` is None or `tool_dispatcher` is None the call collapses to a
single round-trip — same shape as openai_vlm.query so the adapter is
safe to pair with prompts that don't expose tools.

Otherwise: send messages + tools, dispatch every tool_call the model
emits, append the results, loop until the model returns text without
tool_calls or until `max_iters` is hit.

Returned dict (matches the openai_vlm.py contract):
  - "text": final assistant text
  - "usage": per-call token usage SUMMED across turns (incl. cached +
             reasoning where present)
  - "finish_reason": finish_reason of the FINAL turn (the one that
                     returned text, or the last one if max_iters hit)
  - "per_turn_finish_reasons": list, one entry per API round-trip
  - "model_resolved": resolved model snapshot (last turn)
  - "system_fingerprint": last turn's fingerprint
  - "request_id": last turn's chatcmpl-... id
  - "n_turns": number of API round-trips
  - "conversation": full post-loop msgs trace with image data redacted
                    (so reviewers can see the assistant's content
                    alongside its tool_calls without bloating the JSONL
                    with megabytes of base64)

Optional instrumentation: pass dispatcher_kwargs={"_tool_log": list}
and the adapter will append one record per tool call to that list, with
the tool name, raw arguments, and a 200-char preview of the dispatcher
result. The runner uses this to fill the tool_calls / tool_call_log
columns in per_question.csv.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


NAME = "openai_vlm_tools"
SUPPORTS_TOOLS = True


def _usage_to_dict(usage) -> Optional[Dict[str, Any]]:
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


def _add_usage(acc: Dict[str, Any],
               new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum scalar token counts; recurse into nested *_details dicts.

    OpenAI returns prompt_tokens_details.cached_tokens and
    completion_tokens_details.reasoning_tokens as nested dicts; this
    walks them so cached/reasoning totals also accumulate per-turn.
    """
    if not new:
        return acc
    for k, v in new.items():
        if isinstance(v, dict):
            acc[k] = _add_usage(acc.get(k, {}), v)
        elif isinstance(v, (int, float)):
            acc[k] = acc.get(k, 0) + v
        else:
            acc.setdefault(k, v)
    return acc


def _strip_image_data(msg: Any) -> Any:
    """Replace any inline base64 image_url payloads in a message with a
    short placeholder so the conversation trace is readable in JSONL."""
    if not isinstance(msg, dict):
        return msg
    content = msg.get("content")
    if isinstance(content, list):
        new_content = []
        for part in content:
            if (isinstance(part, dict)
                    and part.get("type") == "image_url"):
                url = (part.get("image_url") or {}).get("url", "")
                new_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"<image: {len(url)} chars>"},
                })
            else:
                new_content.append(part)
        return {**msg, "content": new_content}
    return msg


def _serialise_assistant(msg) -> Dict[str, Any]:
    """Convert an OpenAI ChatCompletion message into the dict the API
    expects when the conversation continues. Falls back to manual
    construction on SDKs that lack model_dump."""
    try:
        out = msg.model_dump(exclude_none=True)
    except Exception:
        out = {"role": "assistant", "content": msg.content}
        if getattr(msg, "tool_calls", None):
            out["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
    out["role"] = "assistant"
    return out


def query(messages: List[Dict[str, Any]], *,
          model_id: str = "gpt-5",
          tools: Optional[List[Dict[str, Any]]] = None,
          tool_dispatcher: Optional[Callable[..., str]] = None,
          dispatcher_kwargs: Optional[Dict[str, Any]] = None,
          max_iters: int = 10) -> Dict[str, Any]:
    """Run the tool-using loop and return text + capture metadata."""
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

    # Pull instrumentation kwargs aside so they don't leak into the
    # dispatcher's signature. Dispatchers receive only the keys they
    # explicitly accept (e.g. instance, solution).
    dispatcher_kwargs = dict(dispatcher_kwargs or {})
    tool_log: Optional[list] = dispatcher_kwargs.pop("_tool_log", None)

    # Aggregators populated regardless of single-shot vs loop path so
    # the returned dict always has the same shape.
    total_usage: Dict[str, Any] = {}
    per_turn_finish: List[str] = []
    last_resp = None

    # Single-shot path: prompt didn't offer tools, or adapter wasn't
    # given a dispatcher. Behave like the plain adapter.
    if not tools or tool_dispatcher is None:
        resp = client.chat.completions.create(
            model=model_id, messages=messages,
        )
        choice = resp.choices[0]
        per_turn_finish.append(choice.finish_reason)
        _add_usage(total_usage, _usage_to_dict(resp.usage))
        return {
            "text": choice.message.content or "",
            "usage": total_usage or None,
            "finish_reason": choice.finish_reason,
            "model_resolved": getattr(resp, "model", None),
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "request_id": getattr(resp, "id", None),
            "n_turns": 1,
            "per_turn_finish_reasons": per_turn_finish,
            "conversation": None,
        }

    # Tool-using loop.
    msgs = list(messages)
    last_msg = None
    final_text: Optional[str] = None
    n_turns = 0
    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model=model_id, messages=msgs, tools=tools,
        )
        last_resp = resp
        n_turns += 1
        choice = resp.choices[0]
        per_turn_finish.append(choice.finish_reason)
        _add_usage(total_usage, _usage_to_dict(resp.usage))

        last_msg = choice.message
        msgs.append(_serialise_assistant(last_msg))

        # Final answer — no more tool calls.
        if not last_msg.tool_calls:
            final_text = last_msg.content or ""
            break

        # Execute each tool call and append its result as a `tool` msg.
        for tc in last_msg.tool_calls:
            try:
                result = tool_dispatcher(
                    tc.function.name,
                    tc.function.arguments,
                    **dispatcher_kwargs,
                )
                if not isinstance(result, str):
                    # Be defensive — some dispatchers might return objects.
                    import json as _json
                    result = _json.dumps(result, default=str)
            except Exception as e:
                result = (f'{{"error": "dispatcher raised '
                          f'{type(e).__name__}: {e}"}}')

            if tool_log is not None:
                tool_log.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                    "result_preview": result[:200],
                })

            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Iteration budget exhausted (or loop finished). Compose the final
    # text — fall back to whatever was on the last assistant msg, or a
    # sentinel so the task scorer at least has something to parse.
    if final_text is None:
        final_text = ((last_msg.content if last_msg else "")
                      or "(max_iters reached without final answer)")

    # Build a redacted conversation trace for the JSONL — strip inline
    # base64 image content so the file stays small while preserving the
    # assistant's narration around tool_calls.
    conversation = [_strip_image_data(m) for m in msgs]

    return {
        "text": final_text,
        "usage": total_usage or None,
        "finish_reason": (per_turn_finish[-1] if per_turn_finish else None),
        "model_resolved": (getattr(last_resp, "model", None)
                           if last_resp else None),
        "system_fingerprint": (getattr(last_resp, "system_fingerprint", None)
                               if last_resp else None),
        "request_id": (getattr(last_resp, "id", None)
                       if last_resp else None),
        "n_turns": n_turns,
        "per_turn_finish_reasons": per_turn_finish,
        "conversation": conversation,
    }
