package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

func TestNativeCodexResponsesPreservesClientPayloadAndStream(t *testing.T) {
	requestBody := `{"model":"gpt-5-codex","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"use the shell"}]}],"stream":true,"reasoning":{"effort":"high","summary":"auto"},"text":{"verbosity":"low"},"include":["reasoning.encrypted_content"],"prompt_cache_key":"cache-1","client_metadata":{"originator":"codex_cli_rs"},"additional_tools":[{"type":"custom","name":"apply_patch"}]}`
	var upstreamBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/backend-api/codex/responses" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer codex-token" {
			t.Errorf("authorization = %q", got)
		}
		if got := r.Header.Get("chatgpt-account-id"); got != "account-1" {
			t.Errorf("account header = %q", got)
		}
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"status\":\"in_progress\",\"model\":\"gpt-5-codex\",\"output\":[]}}\n\n")
		io.WriteString(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n\n")
		io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"status\":\"completed\",\"model\":\"gpt-5-codex\",\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"done\"}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"total_tokens\":12}}}\n\n")
	}))
	defer upstream.Close()

	p := NewProxy()
	p.AddProvider(&providers.Provider{
		Name:    "codex",
		Type:    providers.ProviderTypeCodex,
		BaseURL: upstream.URL + "/backend-api/codex",
		Models:  []string{"gpt-5-codex"},
		Auth:    staticBearer{token: "codex-token", headers: map[string]string{"chatgpt-account-id": "account-1"}},
	})
	var observed *ResponseContext
	p.OnResponse(func(ctx *ResponseContext) { observed = ctx })

	w := httptest.NewRecorder()
	p.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(requestBody)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	assertJSONEqual(t, upstreamBody, []byte(requestBody))
	if !strings.Contains(w.Body.String(), "response.completed") {
		t.Fatalf("stream missing terminal event: %s", w.Body.String())
	}
	if observed == nil || observed.Output == nil || observed.Output.Usage == nil || observed.Output.Usage.TotalTokens != 12 {
		t.Fatalf("observed response = %#v", observed)
	}
}

func TestNativeAnthropicMessagesPreservesClaudeCodeExtensions(t *testing.T) {
	requestBody := `{"model":"claude-sonnet-4-5","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"inspect the repository","cache_control":{"type":"ephemeral"}}]}],"system":[{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}}],"tools":[{"name":"Read","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],"thinking":{"type":"adaptive"},"context_management":{"edits":[{"type":"clear_tool_uses_20250919"}]},"output_config":{"effort":"high"},"metadata":{"user_id":"tool-user"}}`
	var upstreamBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer claude-token" {
			t.Errorf("authorization = %q", got)
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Errorf("anthropic-version = %q", got)
		}
		if got := r.Header.Get("anthropic-beta"); !strings.Contains(got, "oauth-2025-04-20") {
			t.Errorf("anthropic-beta = %q", got)
		}
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}`)
	}))
	defer upstream.Close()

	p := NewProxy()
	p.AddProvider(&providers.Provider{
		Name:    "claude",
		Type:    providers.ProviderTypeAnthropic,
		BaseURL: upstream.URL,
		Models:  []string{"claude-sonnet-4-5"},
		Auth:    staticBearer{token: "claude-token"},
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(requestBody))
	req.Header.Set("anthropic-beta", "oauth-2025-04-20,interleaved-thinking-2025-05-14")
	p.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	assertJSONEqual(t, upstreamBody, []byte(requestBody))
}

func TestNativeOpenAIChatPreservesCompatibleProviderExtensions(t *testing.T) {
	requestBody := `{"model":"deepseek-reasoner","messages":[{"role":"developer","content":"be precise"},{"role":"user","content":"hello"}],"stream":false,"thinking":{"type":"enabled"},"provider_options":{"routing":"fast"}}`
	var upstreamBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"chat_1","object":"chat.completion","created":1,"model":"deepseek-reasoner","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	}))
	defer upstream.Close()

	p := NewProxy()
	p.AddProvider(&providers.Provider{Name: "deepseek", BaseURL: upstream.URL + "/v1", APIKey: "key", Models: []string{"deepseek-reasoner"}})
	w := httptest.NewRecorder()
	p.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(requestBody)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	assertJSONEqual(t, upstreamBody, []byte(requestBody))
}

func TestNativeProviderErrorIsPassedThrough(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		io.WriteString(w, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
	}))
	defer upstream.Close()
	p := NewProxy()
	p.AddProvider(&providers.Provider{Name: "claude", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, APIKey: "key", Models: []string{"claude"}})
	w := httptest.NewRecorder()
	p.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"claude","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)))
	if w.Code != http.StatusTooManyRequests || !strings.Contains(w.Body.String(), "rate_limit_error") {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
}

func TestNativeAnthropicStreamIsObserved(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude\",\"content\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n")
		io.WriteString(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		io.WriteString(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"observed\"}}\n\n")
		io.WriteString(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
		io.WriteString(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n")
		io.WriteString(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer upstream.Close()

	p := NewProxy()
	p.AddProvider(&providers.Provider{Name: "claude", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, APIKey: "key", Models: []string{"claude"}})
	var observed *ResponseContext
	p.OnResponse(func(ctx *ResponseContext) { observed = ctx })
	w := httptest.NewRecorder()
	p.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"claude","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if observed == nil || observed.Output == nil || observed.Output.GetMessage().Content != "observed" {
		t.Fatalf("observed response = %#v", observed)
	}
	if observed.Output.Usage == nil || observed.Output.Usage.TotalTokens != 7 {
		t.Fatalf("observed usage = %#v", observed.Output.Usage)
	}
}

func TestProviderModelAliasRewritesOnlyModel(t *testing.T) {
	var upstreamBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		io.WriteString(w, `{"id":"chat_1","object":"chat.completion","created":1,"model":"upstream-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()
	p := NewProxy()
	p.AddProvider(&providers.Provider{
		Name: "alias", BaseURL: upstream.URL + "/v1", APIKey: "key",
		ModelAliases: map[string]string{"public-model": "upstream-model"},
	})
	w := httptest.NewRecorder()
	body := `{"model":"public-model","messages":[{"role":"user","content":"hi"}],"provider_extension":{"keep":true}}`
	p.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var got map[string]any
	if err := json.Unmarshal(upstreamBody, &got); err != nil {
		t.Fatal(err)
	}
	if got["model"] != "upstream-model" {
		t.Fatalf("upstream model = %#v", got["model"])
	}
	if extension, ok := got["provider_extension"].(map[string]any); !ok || extension["keep"] != true {
		t.Fatalf("extension lost: %#v", got)
	}
}

func TestCrossProtocolWarningsReachOnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"id":"chat_1","object":"chat.completion","model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()

	p := NewProxy()
	p.AddProvider(&providers.Provider{Name: "openai", BaseURL: upstream.URL + "/v1", APIKey: "key", Models: []string{"model"}})
	var observed *ResponseContext
	p.OnResponse(func(ctx *ResponseContext) { observed = ctx })
	body := `{"model":"model","max_tokens":128,"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"metadata":{"user_id":"claude-code"}}`
	w := httptest.NewRecorder()
	p.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if observed == nil || len(observed.Diagnostics) < 2 {
		t.Fatalf("response diagnostics = %#v", observed)
	}
	joined := strings.Join(observed.Diagnostics, "\n")
	for _, want := range []string{"[warning] $.metadata", "[warning] $.messages[0].content[0].cache_control"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing %q in diagnostics:\n%s", want, joined)
		}
	}
}

type staticBearer struct {
	token   string
	headers map[string]string
}

func (s staticBearer) BearerToken() (string, map[string]string, error) {
	return s.token, s.headers, nil
}

func assertJSONEqual(t *testing.T, got, want []byte) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatalf("invalid got JSON: %v\n%s", err, got)
	}
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatalf("invalid want JSON: %v\n%s", err, want)
	}
	gotJSON, _ := json.Marshal(gotValue)
	wantJSON, _ := json.Marshal(wantValue)
	if !bytes.Equal(gotJSON, wantJSON) {
		t.Fatalf("JSON differs\ngot:  %s\nwant: %s", gotJSON, wantJSON)
	}
}
