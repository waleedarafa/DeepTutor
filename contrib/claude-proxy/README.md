# DeepTutor with `claude -p` (no API key)

Run DeepTutor's LLM through a local [Claude Code](https://claude.com/claude-code)
subscription via the `claude -p` CLI instead of a paid API key. Embeddings run
locally on [Ollama](https://ollama.com); web search can default to DuckDuckGo
(no key). Net result: a fully working DeepTutor with **zero cloud API keys**.

```
DeepTutor ──/v1/chat/completions──▶ claude_proxy.py ──subprocess──▶ claude -p
          ──/api/embed────────────▶ Ollama (nomic-embed-text, 768-dim)
```

## Why a proxy?

DeepTutor talks to OpenAI-compatible endpoints, and its agent loop drives the
model with a `` ``FINISH``/``TOOL``/``THINK`` `` label protocol plus OpenAI
function tool-calls. The `claude` CLI can't emit native tool-calls, so
`claude_proxy.py`:

- exposes `/v1/chat/completions` (streaming + non-streaming) and `/v1/models`;
- maps each request to one `claude -p --output-format json` call;
- replaces Claude Code's heavy coding-agent system prompt with `--system-prompt`,
  while forwarding DeepTutor's own agent prompt verbatim in agentic mode;
- instructs the model to emit a JSON `tool_calls` block after the `` ``TOOL`` ``
  label and converts it back into OpenAI `tool_calls` — so RAG and the rest of
  the agentic tool loop work transparently.

It is **stdlib-only** (no third-party deps) and runs under any Python 3.11+.

## Prerequisites

- DeepTutor installed (`pip install -U deeptutor`).
- [Claude Code](https://claude.com/claude-code) logged in: `claude -p "hi"` works.
- Ollama with an embedding model: `ollama pull nomic-embed-text`.

## Setup

1. **Start the proxy** (keep it running while you use DeepTutor):
   ```bash
   ./start_proxy.sh                      # default model: haiku (fast/cheap)
   CLAUDE_PROXY_MODEL=sonnet ./start_proxy.sh   # higher quality
   CLAUDE_PROXY_MODEL=opus   ./start_proxy.sh   # best quality, priciest
   ```

2. **Point DeepTutor at it.** Run `deeptutor init` and choose the *Custom /
   Other* LLM provider with base URL `http://127.0.0.1:8088/v1`, any api key
   (e.g. `sk-local`), model `claude-code`; and the *Ollama* embedding provider
   with `nomic-embed-text`. Or copy the ready-made catalog:
   ```bash
   cp model_catalog.sample.json <DEEPTUTOR_HOME>/data/user/settings/model_catalog.json
   ```
   Verify with `deeptutor config show`.

3. **Use it:**
   ```bash
   deeptutor kb create my-book --doc /path/to/book.pdf      # ingest (embeddings only)
   deeptutor run chat "Summarize chapter 1" -t rag --kb my-book
   deeptutor chat --kb my-book                              # interactive tutor
   deeptutor start                                          # full web UI on :3782
   ```

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_PROXY_MODEL` | `haiku` | Model alias passed to `claude -p` (`haiku`/`sonnet`/`opus`). |
| `CLAUDE_PROXY_PORT` | `8088` | Bind port. |
| `CLAUDE_PROXY_HOST` | `127.0.0.1` | Bind host. |
| `CLAUDE_BIN` | (PATH) | Path to the `claude` executable. |
| `CLAUDE_PROXY_TIMEOUT` | `600` | Per-request subprocess timeout (seconds). |
| `CLAUDE_PROXY_DEBUG` | (unset) | When set, dumps each request to `proxy_requests.log`. |

## Notes & limitations

- Each tutoring turn is a 2–3 round-trip agent loop through `claude -p`, so
  expect ~15–30s per answer on `haiku`; higher-tier models trade speed for depth.
- Embeddings never touch Claude — ingestion is local and free.
- Cost figures printed by `claude -p` are notional under a Claude Code subscription.
- Tested against DeepTutor v1.4.1 (`chat` capability with the `rag` tool).
