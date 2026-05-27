#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim that backends onto the `claude -p` CLI.

DeepTutor (and any OpenAI SDK client) can point its LLM `base_url` at this
server. Each `/v1/chat/completions` request is translated into a single
`claude -p --output-format json` subprocess call, so generation runs on the
local Claude Code subscription instead of a metered API key.

Stdlib only (no third-party deps) so it runs under any interpreter.

Config via environment:
  CLAUDE_PROXY_HOST   bind host          (default 127.0.0.1)
  CLAUDE_PROXY_PORT   bind port          (default 8088)
  CLAUDE_PROXY_MODEL  model alias        (default sonnet; e.g. haiku/opus)
  CLAUDE_BIN          claude executable  (default: resolved from PATH)
  CLAUDE_PROXY_TIMEOUT subprocess timeout seconds (default 600)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOG = logging.getLogger("claude_proxy")

HOST = os.environ.get("CLAUDE_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLAUDE_PROXY_PORT", "8088"))
MODEL = os.environ.get("CLAUDE_PROXY_MODEL", "sonnet")
CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"
TIMEOUT_S = int(os.environ.get("CLAUDE_PROXY_TIMEOUT", "600"))
MODEL_ID = "claude-code"

DEFAULT_SYSTEM = (
    "You are a knowledgeable, precise tutor. Answer the user's question "
    "clearly and concisely. When context is provided, ground your answer in "
    "it and do not invent facts beyond it."
)

# Appended only in agentic (tools present) mode. DeepTutor's own system prompt
# already defines the ``FINISH``/``TOOL``/``THINK`` label protocol; this endpoint
# cannot emit native tool_calls, so we describe a JSON convention instead and
# render the tool schemas (which otherwise only arrive via the OpenAI `tools`
# field) as text the model can read.
TOOL_SHIM = (
    "\n\n## Tool-calling on this endpoint\n"
    "This endpoint does NOT support native tool_calls. So when your chosen "
    "action is ``TOOL``, format the reply EXACTLY like this:\n"
    "``TOOL``\n"
    "<one short sentence of intent>\n"
    "```json\n"
    '{"tool_calls": [{"name": "<tool_name>", "arguments": { ... }}]}\n'
    "```\n"
    "Emit one object per tool you want to call inside the `tool_calls` array. "
    "Use only the tools and argument names listed below. For ``FINISH``, "
    "``THINK``, and ``PAUSE`` replies, do NOT emit any JSON — just the label "
    "and your text as instructed above.\n\n"
    "Available tools:\n"
)


def _text_from_content(content: Any) -> str:
    """Extract plain text from an OpenAI message `content` (str or parts list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _render_turn(msg: dict[str, Any]) -> str | None:
    """Render one non-system OpenAI message into a transcript line."""
    role = str(msg.get("role") or "user")
    text = _text_from_content(msg.get("content")).strip()

    if role == "assistant" and msg.get("tool_calls"):
        calls = []
        for tc in msg["tool_calls"]:
            fn = (tc or {}).get("function") or {}
            calls.append(f'{fn.get("name", "?")}({fn.get("arguments", "{}")})')
        body = "; ".join(calls)
        return f"Assistant (called tool): {body}" + (f"\n{text}" if text else "")

    if role == "tool":
        name = msg.get("name") or msg.get("tool_call_id") or "tool"
        return f"Tool result from {name}:\n{text}"

    if not text:
        return None
    labels = {"user": "User", "assistant": "Assistant"}
    return f"{labels.get(role, role.title())}: {text}"


def _tools_instruction(tools: list[dict[str, Any]] | None) -> str:
    """Render an OpenAI `tools` array into a plain-text protocol block."""
    if not tools:
        return ""
    lines: list[str] = []
    for tool in tools:
        fn = (tool or {}).get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        desc = (fn.get("description") or "").strip()
        params = fn.get("parameters") or {}
        lines.append(f"- {name}: {desc}\n  parameters (JSON schema): {json.dumps(params)}")
    return TOOL_SHIM + "\n".join(lines) if lines else ""


def _split_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, conversation_transcript)."""
    system_chunks: list[str] = []
    rendered: list[str] = []
    for msg in messages or []:
        if str(msg.get("role")) == "system":
            text = _text_from_content(msg.get("content")).strip()
            if text:
                system_chunks.append(text)
            continue
        line = _render_turn(msg)
        if line:
            rendered.append(line)

    # In agentic mode DeepTutor's own system prompt is authoritative (it defines
    # the label protocol); don't shadow it with our tutor preamble. In plain
    # chat mode, fall back to our tutor persona.
    if tools:
        system_prompt = "\n\n".join(system_chunks) if system_chunks else DEFAULT_SYSTEM
        system_prompt += _tools_instruction(tools)
    elif system_chunks:
        system_prompt = DEFAULT_SYSTEM + "\n\n" + "\n\n".join(system_chunks)
    else:
        system_prompt = DEFAULT_SYSTEM

    # Single bare user turn stays bare; otherwise a labelled transcript that
    # also carries prior tool calls/results for the agent loop.
    if len(rendered) == 1 and rendered[0].startswith("User: "):
        transcript = rendered[0][len("User: "):]
    else:
        transcript = "\n\n".join(rendered)
    return system_prompt, transcript


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_TOOL_LABEL_RE = re.compile(r"^\s*`{1,3}\s*TOOL\s*`{1,3}", re.IGNORECASE)


def _is_tool_label(text: str) -> bool:
    """True when the reply opens with the ``TOOL`` protocol label."""
    return bool(_TOOL_LABEL_RE.match(text or ""))


def _normalize_calls(obj: Any) -> list[dict[str, Any]]:
    """Coerce parsed JSON into a list of {name, arguments} tool calls."""
    candidates: list[Any] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("tool_calls"), list):
            candidates = obj["tool_calls"]
        elif isinstance(obj.get("tool_call"), dict):
            candidates = [obj["tool_call"]]
        elif obj.get("name"):
            candidates = [obj]
    elif isinstance(obj, list):
        candidates = obj

    calls: list[dict[str, Any]] = []
    for cand in candidates:
        if isinstance(cand, dict) and cand.get("name"):
            args = cand.get("arguments")
            calls.append({
                "name": str(cand["name"]),
                "arguments": args if isinstance(args, dict) else {},
            })
    return calls


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool call(s) from a reply that may carry a label + fenced JSON."""
    # Prefer a fenced ```json block if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    blob = fence.group(1) if fence else None
    if blob is None:
        # Fall back to the first/last balanced-looking JSON object in the text.
        match = _JSON_OBJ_RE.search(text)
        blob = match.group(0) if match else None
    if not blob:
        return []
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return []
    return _normalize_calls(obj)


def _call_claude(system_prompt: str, transcript: str) -> str:
    """Run a single `claude -p` turn and return the assistant text."""
    args = [
        CLAUDE_BIN,
        "-p",
        "--output-format", "json",
        "--model", MODEL,
        "--system-prompt", system_prompt,
        "--no-session-persistence",
    ]
    proc = subprocess.run(
        args,
        input=transcript,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()[:500]}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"claude returned non-JSON: {proc.stdout.strip()[:500]}") from exc
    if payload.get("is_error"):
        raise RuntimeError(f"claude reported error: {payload.get('result') or payload}")
    result = payload.get("result")
    if not isinstance(result, str):
        raise RuntimeError(f"claude JSON missing string 'result': {payload}")
    return result


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Quieter access logging routed through our logger.
    def log_message(self, fmt: str, *fmt_args: Any) -> None:  # noqa: N802
        LOG.info("%s - %s", self.address_string(), fmt % fmt_args)

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/v1/models") or self.path.rstrip("/") == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "anthropic"}],
            })
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": {"message": f"bad request body: {exc}"}})
            return

        messages = req.get("messages") or []
        tools = req.get("tools")
        stream = bool(req.get("stream"))
        if os.environ.get("CLAUDE_PROXY_DEBUG"):
            with open("proxy_requests.log", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(req, indent=2) + "\n=====\n")
        system_prompt, transcript = _split_messages(messages, tools)

        started = time.time()
        try:
            content = _call_claude(system_prompt, transcript)
        except Exception as exc:  # surface a clean OpenAI-style error
            LOG.error("claude call failed: %s", exc)
            self._send_json(502, {"error": {"message": str(exc), "type": "upstream_error"}})
            return

        tool_calls = _parse_tool_calls(content) if (tools and _is_tool_label(content)) else []
        LOG.info("served chat (%d msgs, tools=%d, stream=%s) in %.1fs -> %s",
                 len(messages), len(tools or []), stream, time.time() - started,
                 f"tool_calls:{[c['name'] for c in tool_calls]}" if tool_calls
                 else f"{len(content)} chars")

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        usage = {
            "prompt_tokens": _estimate_tokens(system_prompt + transcript),
            "completion_tokens": _estimate_tokens(content),
            "total_tokens": _estimate_tokens(system_prompt + transcript + content),
        }

        openai_tool_calls = None
        finish_reason = "stop"
        message_content: str | None = content
        if tool_calls:
            finish_reason = "tool_calls"
            # Keep the ``TOOL`` label + intent text as content so DeepTutor's
            # label parser still sees a coherent reply; tool_calls are
            # authoritative for action resolution.
            openai_tool_calls = [{
                "id": f"call_{uuid.uuid4().hex[:9]}",
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c["arguments"]),
                },
            } for c in tool_calls]

        if stream:
            self._send_stream(completion_id, created, message_content,
                              usage, finish_reason, openai_tool_calls)
        else:
            message: dict[str, Any] = {"role": "assistant", "content": message_content}
            if openai_tool_calls:
                message["tool_calls"] = openai_tool_calls
            self._send_json(200, {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }],
                "usage": usage,
            })

    def _send_stream(self, cid: str, created: int, content: str | None,
                     usage: dict[str, int], finish_reason: str = "stop",
                     tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def chunk(delta: dict[str, Any], finish: str | None = None,
                  extra: dict[str, Any] | None = None) -> None:
            payload: dict[str, Any] = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if extra:
                payload.update(extra)
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))

        chunk({"role": "assistant"})
        if tool_calls:
            # Emit each tool call as an indexed delta (OpenAI streaming shape).
            for idx, tc in enumerate(tool_calls):
                chunk({"tool_calls": [{
                    "index": idx,
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }]})
        elif content:
            step = 512
            for i in range(0, len(content), step):
                chunk({"content": content[i:i + step]})
        chunk({}, finish=finish_reason, extra={"usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOG.info("claude-proxy starting on http://%s:%d (model=%s, bin=%s)",
             HOST, PORT, MODEL, CLAUDE_BIN)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
