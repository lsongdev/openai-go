# Proxy Architecture

Status: implemented current architecture

Updated: 2026-08-23

## Decision

`cf-workers-openai` has the better protocol architecture: a gateway, provider
adapters, a protocol kit, and a canonical semantic model used only when two
wire protocols differ. The former `router/` in this repository had useful Go
clients, but used OpenAI Chat Completions structs as its implicit canonical
model. That loses Responses items, Anthropic content blocks, provider-specific
fields, and stream state.

This repository adopts the reference architecture, but not its Worker-specific
billing, D1, or lifecycle code:

1. **Gateway (`proxy.Proxy`)** owns endpoints, admission hooks, model routing,
   provider selection, and downstream error envelopes.
2. **Provider adapters (`proxy/providers`)** own native protocol, endpoint,
   authentication, required headers, model aliases, and provider request rules.
3. **Protocol codecs (`proxy/codec`)** own request, response, and SSE semantics
   for OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages.
4. **Canonical types** are used only for cross-protocol conversion and response
   observation. They are not a storage format and are not used on native paths.

This gives the Go service the same protocol boundaries as the reference project
without importing infrastructure that belongs to an edge billing gateway.

The package layout makes source and protocol ownership explicit:

```text
proxy/
|-- providers/
|   |-- provider.go       shared provider contract and HTTP dispatch
|   |-- openai/           OpenAI-compatible API-key source adapter
|   |-- anthropic/        Anthropic API-key source adapter
|   |-- codex/            ChatGPT Codex OAuth source adapter
|   `-- claudecode/       Claude Code OAuth source adapter
|-- protocols/            /chat/completions, /responses, /messages entrypoints
`-- codec/                request, response, and SSE protocol transforms
```

The repository-level `openai/` and `anthropic/` packages own the reusable API
clients and wire types. Their low-level `NewRequest` and `Do` methods preserve
raw request bodies, status codes, response headers, response bodies, and SSE
streams. Proxy source adapters configure and call those clients; they do not
duplicate client transports or protocol models.

## Request Flow

The public endpoints identify the client protocol:

| Endpoint | Client protocol |
| --- | --- |
| `/v1/chat/completions` | `openai.chat.v1` |
| `/v1/responses` | `openai.responses.v1` |
| `/v1/messages` | `anthropic.messages.v1` |

Each handler decodes only `model` and `stream` before routing. If the selected
provider has the same native protocol, the original JSON is forwarded without
decoding and re-encoding it. A configured public model alias may replace only
the top-level `model` value.

If the protocols differ, the request follows this pipeline:

```text
client JSON -> source codec -> canonical request -> target codec -> provider
provider JSON/SSE -> source codec/state machine -> canonical response/events
                  -> target codec/state machine -> client
```

Three protocols therefore require three codecs rather than six pairwise
converters. Adding a fourth protocol requires one decoder/encoder pair and one
stream state machine.

## Provider Model

Provider `Source` selects authentication and provider behavior. `Protocol`
selects the native wire format. They are deliberately separate: `codex` is an
OAuth/provider adapter whose native wire format is OpenAI Responses.

When `protocol` is omitted it is inferred as follows:

| Provider source | Native protocol |
| --- | --- |
| `openai` or empty | `openai.chat.v1` |
| `anthropic` | `anthropic.messages.v1` |
| `codex` | `openai.responses.v1` |
| `claudecode` | `anthropic.messages.v1` |

Configured API-key providers, local Codex credentials in `~/.codex/auth.json`,
and local Claude Code credentials in `~/.claude/.credentials.json` can coexist.
OAuth credentials are resolved per request so token refresh does not require
restarting the proxy. The `codex` and `claudecode` sources are registered from
their local login stores; JSON-configured providers currently accept `openai`
and `anthropic` sources.

The `models` configuration maps a public model name to an upstream model ID:

```json
{
  "providers": {
    "Claude API": {
      "type": "anthropic",
      "protocol": "anthropic.messages.v1",
      "apiBase": "https://api.anthropic.com/v1",
      "apiKey": "...",
      "models": {
        "claude": "claude-sonnet-4-6"
      }
    }
  }
}
```

Routing uses the public model name. `X-Miya-Provider` can explicitly select a
provider when two providers expose the same model or a client cannot change its
model value. The header is consumed by the gateway and is not forwarded.
Provider names and model aliases are case-sensitive. Without an explicit
header, duplicate public model names resolve to the lexicographically first
provider; `/v1/models` uses the same deterministic ownership and de-duplicates
public names.

The Codex adapter additionally enforces the ChatGPT backend requirements
`store=false` and `stream=true`, and removes unsupported
`max_output_tokens`, but only for cross-protocol requests. Native Codex payloads
remain untouched. Codex model metadata is loaded from
`~/.codex/models_cache.json` when available so the CLI can discover model
capabilities through `/v1/models`.

## Conversion Contract

The currently portable canonical subset is intentionally smaller than the
union of all provider APIs:

- system, developer, user, assistant, and tool conversation turns;
- text and reasoning summary output;
- function tool definitions, choices, calls, fragmented JSON arguments, tool
  results, and error results;
- maximum output tokens, temperature, top-p, stop sequences, and streaming;
- stop reason, input/output totals, cached input, cache creation, and reasoning
  token counts when the source protocol supplies them.

Native traffic preserves unknown fields. Cross-protocol traffic rejects unknown
or unsupported semantic data with a `codec.TranslationError` before calling the
provider. Documented provider execution hints may be dropped with warning
diagnostics exposed to `OnResponse`; examples include prompt cache directives,
client metadata, and provider-specific thinking/output controls.

The following are not currently portable and are rejected on cross-protocol
paths:

- images, audio, documents, and other media blocks;
- citations, annotations with semantics, and structured output schemas;
- Responses built-in/custom tools such as computer use, web search, file
  search, code interpreter, and Codex `additional_tools`;
- stateful Responses behavior such as `previous_response_id` when it cannot be
  represented by the target protocol;
- background request semantics, provider container state, and remote MCP
  execution; response-only provider metadata may instead produce a warning;
- refusal content when the target would represent it only as ordinary text;
- signed or encrypted reasoning round-trips.

Reasoning summaries can be displayed across protocols, but Anthropic thinking
signatures and Responses encrypted reasoning items are not equivalent. They are
not claimed as lossless reasoning history support.

## Streaming

Streaming conversion is stateful:

```text
bytes -> SSE frames -> source events -> canonical events -> target events
```

The SSE reader supports CRLF, arbitrary reader chunk boundaries, multi-line
`data`, comments, IDs, retry fields, EOF dispatch, cancellation, and upstream
error bodies. Protocol state machines assemble text, reasoning, multiple tool
calls, fragmented tool JSON, terminal reasons, and final usage.

If a provider always streams while the client requested JSON, the proxy
collects the stream into a canonical response. If a provider returns JSON while
the client requested a stream, the proxy synthesizes a protocol-correct stream.

OpenAI Responses reports usage at its terminal event, while Anthropic normally
places input usage in `message_start`. During Responses-to-Anthropic streaming,
the proxy starts output immediately and reports final usage when it becomes
available; it does not delay the first token merely to know input usage.

## Hooks And Errors

`OnRequest` receives the raw request, routing fields, parsed best-effort Chat
view, and selected provider. It may reject the request or override routing.

`OnResponse` receives the assembled best-effort Chat view, duration, terminal
error, and conversion diagnostics. Hook parsing never changes native payloads.
Native provider response bodies and errors are passed through to the client.
Cross-protocol errors use the client's error envelope and include diagnostic
paths such as `$.tools[9].type`.

## Verification

Automated tests cover all three native handlers and all cross-protocol text
directions, JSON and SSE responses, fragmented function calls, tool results,
usage observation, native extension preservation, model aliases, upstream
errors, CRLF/multiline SSE, and strict conversion diagnostics.

Real client verification was run locally on 2026-08-23 with Codex CLI 0.147.0
and Claude Code 2.1.146 against real subscription-backed providers:

| Client wire protocol | Provider wire protocol | Text | File tool loop | Result |
| --- | --- | --- | --- | --- |
| Codex / Responses | Codex / Responses | pass | pass | native payload and stream pass-through |
| Claude Code / Messages | Claude / Messages | pass | pass | native Claude extensions preserved |
| Claude Code / Messages | Codex / Responses | pass | pass | request, SSE, tool use/result converted |
| Codex / Responses | Claude / Messages | rejected | rejected | expected: Codex emits Responses-only controls and `additional_tools` that cannot be represented |

The successful tool scenarios created a file through the real client tool,
read it back, sent the tool result to the model, and received a required final
marker. The rejected Codex-to-Claude case failed locally before any provider
request. Codex 0.147.0 reported unsupported paths for `include`,
`parallel_tool_calls`, `reasoning`, `text`, and an `additional_tools` input
item. Silently deleting those fields would make the matrix look greener while
changing client behavior, so that combination remains explicitly unsupported.

Claude Code only honored the proxy base URL when supplied through its settings
environment, for example:

```sh
claude --settings '{"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:8090"}}'
```

Provider selection can be supplied to Claude Code in the same settings object:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8090",
    "ANTHROPIC_CUSTOM_HEADERS": "X-Miya-Provider: claudecode"
  }
}
```

Live tests consume provider tokens and are intentionally not part of the
default Go test suite.

## Remaining Boundaries

Future protocol work should extend the canonical model only when end-to-end
semantics and fixtures exist. The next useful additions are media content,
structured output, refusal content, built-in tool capability negotiation, and
signed reasoning history. Stateful Responses features require a gateway state
mapping, not just another JSON field.

For production hardening, add fuzz and load tests for oversized SSE events,
half-closed clients, backpressure, duplicate terminal events, and providers
that send a terminal event but leave the HTTP connection open.
