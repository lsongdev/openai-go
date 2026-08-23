package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lsongdev/miya-agents/anthropic"
	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy/providers"
)

func TestOnRequestHookRejection(t *testing.T) {
	r := NewProxy()
	r.AddProvider(&providers.Provider{
		Name:    "test",
		Source:  providers.SourceOpenAI,
		BaseURL: "http://localhost",
		Models:  []string{"test-model"},
	})

	r.OnRequest(func(ctx *providers.RequestContext) error {
		return fmt.Errorf("insufficient balance")
	})

	body := `{"model":"test-model","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	r.ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("expected status %d, got %d", http.StatusForbidden, w.Code)
	}
	bodyStr := w.Body.String()
	if !strings.Contains(bodyStr, "insufficient balance") {
		t.Errorf("expected error message to contain 'insufficient balance', got %q", bodyStr)
	}
}

func TestOnRequestHookAllow(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		resp := openai.ChatCompletionResponse{
			ID: "test",
			Choices: []openai.ChatCompletionChoice{
				{Message: &openai.ChatCompletionMessage{Role: "assistant", Content: "hi"}},
			},
			Usage: &openai.CompletionUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{
		Name:    "test",
		Source:  providers.SourceOpenAI,
		BaseURL: upstream.URL,
		Models:  []string{"test-model"},
	})
	var requestCalled bool
	r.OnRequest(func(ctx *providers.RequestContext) error {
		requestCalled = true
		ctx.Upstream = r.FindProviderForModel(ctx.Input.Model)
		return nil
	})

	body := `{"model":"test-model","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	r.ServeHTTP(w, req)

	if !requestCalled {
		t.Error("OnRequest hook was not called")
	}
	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}
}

func TestAnthropicToOpenAIResponse(t *testing.T) {
	anthJSON := `{
		"id": "msg_01abc",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-7-sonnet-20250219",
		"content": [
			{"type": "thinking", "thinking": "Let me think..."},
			{"type": "text", "text": "Hello world"}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5},
		"stop_reason": "end_turn"
	}`
	var authResp anthropic.Response
	json.Unmarshal([]byte(anthJSON), &authResp)
	oaiResp := anthropic.NewChatCompletionResponseFromAnthropicResponse(&authResp)
	if oaiResp.ID != "msg_01abc" {
		t.Errorf("id = %q, want msg_01abc", oaiResp.ID)
	}
	if oaiResp.Choices[0].Message.Content != "Hello world" {
		t.Errorf("content = %q, want Hello world", oaiResp.Choices[0].Message.Content)
	}
	if oaiResp.Choices[0].Message.ReasoningContent != "Let me think..." {
		t.Errorf("reasoning_content = %q, want Let me think...", oaiResp.Choices[0].Message.ReasoningContent)
	}
	if oaiResp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", oaiResp.Choices[0].FinishReason)
	}
	if oaiResp.Usage.PromptTokens != 10 {
		t.Errorf("prompt_tokens = %d, want 10", oaiResp.Usage.PromptTokens)
	}
	if oaiResp.Usage.CompletionTokens != 5 {
		t.Errorf("completion_tokens = %d, want 5", oaiResp.Usage.CompletionTokens)
	}
	if oaiResp.Usage.TotalTokens != 15 {
		t.Errorf("total_tokens = %d, want 15", oaiResp.Usage.TotalTokens)
	}
}

func TestMapAnthropicStopReason(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"end_turn", "stop"},
		{"stop_sequence", "stop"},
		{"max_tokens", "length"},
		{"tool_use", "tool_calls"},
		{"unknown", "unknown"},
	}
	for _, tt := range tests {
		got := anthropic.MapStopReason(tt.input)
		if got != tt.expected {
			t.Errorf("MapStopReason(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestAnthropicEventParsing(t *testing.T) {
	data := `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}`
	var event anthropic.Event
	if err := json.Unmarshal([]byte(data), &event); err != nil {
		t.Fatalf("failed to parse event: %v", err)
	}

	var delta anthropic.Delta
	if err := json.Unmarshal(event.Delta, &delta); err != nil {
		t.Fatalf("failed to parse delta: %v", err)
	}
	if delta.Type != "thinking_delta" {
		t.Errorf("delta.type = %q, want thinking_delta", delta.Type)
	}
	if delta.Thinking != "Let me think" {
		t.Errorf("delta.thinking = %q, want Let me think", delta.Thinking)
	}
}
