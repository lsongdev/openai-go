package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy/providers"
)

type staticTokens struct{ token string }

func (s staticTokens) BearerToken() (string, map[string]string, error) {
	return s.token, nil, nil
}

func TestProxy_Codex_ChatCompletions(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "Bearer tok" {
			t.Errorf("authorization = %q", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openai.ResponseObject{
			ID: "resp_1", Object: "response", CreatedAt: 1, Status: "completed", Model: "gpt-5-codex",
			Output: []openai.ResponseOutputItem{{
				Type: "message", Role: "assistant",
				Content: []openai.ResponseContentPart{{Type: "output_text", Text: "from codex"}},
			}},
			Usage: &openai.ResponseUsage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5},
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	p := codexProvider(t, upstream.URL)
	r.AddProvider(p)

	body := `{"model":"gpt-5-codex","messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp openai.ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if msg := resp.GetMessage(); msg == nil || msg.Content != "from codex" {
		t.Errorf("message = %+v", resp.GetMessage())
	}
}

func TestProxy_Codex_ResponsesStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		created, _ := json.Marshal(map[string]any{
			"type":     "response.created",
			"response": map[string]any{"id": "resp_2", "object": "response", "created_at": 7, "model": "gpt-5-codex", "status": "in_progress"},
		})
		fmt.Fprintf(w, "event: response.created\ndata: %s\n\n", created)
		delta, _ := json.Marshal(map[string]any{"type": "response.output_text.delta", "delta": "streamed"})
		fmt.Fprintf(w, "event: response.output_text.delta\ndata: %s\n\n", delta)
		done, _ := json.Marshal(map[string]any{
			"type":     "response.completed",
			"response": map[string]any{"id": "resp_2", "object": "response", "created_at": 7, "model": "gpt-5-codex", "status": "completed"},
		})
		fmt.Fprintf(w, "event: response.completed\ndata: %s\n\n", done)
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(codexProvider(t, upstream.URL))

	body := `{"model":"gpt-5-codex","input":"hi","stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	out := w.Body.String()
	if !strings.Contains(out, "response.created") || !strings.Contains(out, "output_text.delta") {
		t.Errorf("expected responses SSE events, got: %s", out)
	}
	if !strings.Contains(out, "streamed") {
		t.Errorf("expected delta text in output")
	}
}

func codexProvider(t *testing.T, baseURL string) *providers.Provider {
	t.Helper()
	p := &providers.Provider{
		Name:    "codex-test",
		Type:    providers.ProviderTypeCodex,
		BaseURL: baseURL,
		Models:  []string{"gpt-5-codex"},
		Auth:    staticTokens{token: "tok"},
	}
	return p
}
