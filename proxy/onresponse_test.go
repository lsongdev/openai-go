package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/lsongdev/miya-agents/anthropic"
	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy/providers"
)

// mockResponse captures OnResponse callback data for assertions.
type mockResponse struct {
	mu        sync.Mutex
	called    bool
	requestID string
	input     any
	output    *openai.ChatCompletionResponse
	err       error
}

func (m *mockResponse) capture() func(*providers.ResponseContext) {
	return func(ctx *providers.ResponseContext) {
		m.mu.Lock()
		defer m.mu.Unlock()
		m.called = true
		m.requestID = ctx.RequestID
		m.input = ctx.Input
		m.output = ctx.Output
		m.err = ctx.Error
	}
}

func (m *mockResponse) assertCalled(t *testing.T) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.called {
		t.Fatal("OnResponse was not called")
	}
}

func (m *mockResponse) assertNoError(t *testing.T) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.err != nil {
		t.Fatalf("expected no error, got %v", m.err)
	}
}

func (m *mockResponse) assertUsage(t *testing.T, prompt, completion, total int) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.output == nil || m.output.Usage == nil {
		t.Fatalf("expected usage in output, got nil")
	}
	if m.output.Usage.PromptTokens != prompt {
		t.Errorf("prompt_tokens = %d, want %d", m.output.Usage.PromptTokens, prompt)
	}
	if m.output.Usage.CompletionTokens != completion {
		t.Errorf("completion_tokens = %d, want %d", m.output.Usage.CompletionTokens, completion)
	}
	if m.output.Usage.TotalTokens != total {
		t.Errorf("total_tokens = %d, want %d", m.output.Usage.TotalTokens, total)
	}
}

func (m *mockResponse) assertInputMessage(t *testing.T, expectedContent string) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	switch in := m.input.(type) {
	case *openai.ChatCompletionRequest:
		if len(in.Messages) == 0 {
			t.Fatal("expected input messages, got none")
		}
		if in.Messages[0].Content != expectedContent {
			t.Errorf("input message content = %q, want %q", in.Messages[0].Content, expectedContent)
		}
	default:
		t.Fatalf("unexpected input type: %T", m.input)
	}
}

func (m *mockResponse) assertOutputContent(t *testing.T, expected string) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.output == nil {
		t.Fatal("expected output, got nil")
	}
	if len(m.output.Choices) == 0 {
		t.Fatal("expected output choices, got none")
	}
	if m.output.Choices[0].Message == nil {
		t.Fatal("expected output message, got nil")
	}
	if m.output.Choices[0].Message.Content != expected {
		t.Errorf("output content = %q, want %q", m.output.Choices[0].Message.Content, expected)
	}
}

// ============================================================================
// Mock Tests: All provider/format/stream combinations
// ============================================================================

func TestMock_OpenAI_NonStream_OnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "cmpl-1", Object: "chat.completion", Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{{
				Index: 0, Message: &openai.ChatCompletionMessage{Role: "assistant", Content: "Hello from OpenAI"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		})
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 10, 5, 15)
	mr.assertInputMessage(t, "hi")
	mr.assertOutputContent(t, "Hello from OpenAI")
}

func TestMock_OpenAI_Stream_OnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []openai.ChatCompletionResponse{
			{ID: "cmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Role: "assistant", Content: "Hel"}}}},
			{ID: "cmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Content: "lo"}}}},
			{ID: "cmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{}, FinishReason: "stop"}}, Usage: &openai.CompletionUsage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12}},
		}
		for _, c := range chunks {
			data, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 8, 4, 12)
	mr.assertInputMessage(t, "hi")
	mr.assertOutputContent(t, "Hello")
}

func TestMock_Anthropic_NonStream_OnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(anthropic.Response{
			ID: "msg-1", Type: "message", Model: "claude-3",
			Content: []anthropic.ContentBlock{
				{Type: "thinking", Thinking: "Thinking..."},
				{Type: "text", Text: "Hello from Claude"},
			},
			Usage:      anthropic.Usage{InputTokens: 20, OutputTokens: 10},
			StopReason: "end_turn",
		})
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})
	r.OnResponse(mr.capture())

	body := `{"model":"claude-3","messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 20, 10, 30)
	mr.assertInputMessage(t, "hi")
	mr.assertOutputContent(t, "Hello from Claude")
}

func TestMock_Anthropic_Stream_OnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		msgStart, _ := json.Marshal(anthropic.Event{
			Type: "message_start",
			Message: &anthropic.MessageStart{
				ID: "msg-1", Type: "message", Role: "assistant", Model: "claude-3",
				Usage: anthropic.Usage{InputTokens: 15},
			},
		})
		fmt.Fprintf(w, "event: message_start\ndata: %s\n\n", msgStart)

		deltaData, _ := json.Marshal(anthropic.Delta{Type: "text_delta", Text: "Hello"})
		cbDelta, _ := json.Marshal(anthropic.Event{Type: "content_block_delta", Index: new(int), Delta: deltaData})
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", cbDelta)

		msgDeltaData, _ := json.Marshal(anthropic.Delta{StopReason: "end_turn"})
		msgDelta, _ := json.Marshal(anthropic.Event{Type: "message_delta", Delta: msgDeltaData, Usage: &anthropic.Usage{OutputTokens: 7}})
		fmt.Fprintf(w, "event: message_delta\ndata: %s\n\n", msgDelta)

		msgStop, _ := json.Marshal(anthropic.Event{Type: "message_stop"})
		fmt.Fprintf(w, "event: message_stop\ndata: %s\n\n", msgStop)
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})
	r.OnResponse(mr.capture())

	body := `{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 15, 7, 22)
	mr.assertInputMessage(t, "hi")
	mr.assertOutputContent(t, "Hello")
}

func TestMock_OpenAI_to_Anthropic_NonStream_OnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "cmpl-1", Object: "chat.completion", Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{{
				Index: 0, Message: &openai.ChatCompletionMessage{Role: "assistant", Content: "Hello"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
		})
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var anthResp anthropic.Response
	if err := json.Unmarshal(w.Body.Bytes(), &anthResp); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if anthResp.Content[0].Text != "Hello" {
		t.Errorf("anthropic content = %q, want 'Hello'", anthResp.Content[0].Text)
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 5, 3, 8)
	mr.assertOutputContent(t, "Hello")
}

func TestMock_OpenAI_to_Anthropic_Stream_OnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []openai.ChatCompletionResponse{
			{ID: "cmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Role: "assistant", Content: "Hi"}}}},
			{ID: "cmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{}, FinishReason: "stop"}}, Usage: &openai.CompletionUsage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}},
		}
		for _, c := range chunks {
			data, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	bodyStr := w.Body.String()
	if !strings.Contains(bodyStr, "message_start") {
		t.Errorf("expected message_start in SSE output, got: %s", bodyStr)
	}
	if !strings.Contains(bodyStr, "content_block_delta") {
		t.Errorf("expected content_block_delta in SSE output, got: %s", bodyStr)
	}
	if !strings.Contains(bodyStr, "Hi") {
		t.Errorf("expected 'Hi' in SSE output, got: %s", bodyStr)
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 3, 2, 5)
	mr.assertOutputContent(t, "Hi")
}

func TestMock_Anthropic_to_OpenAI_NonStream_OnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(anthropic.Response{
			ID: "msg-1", Type: "message", Model: "claude-3",
			Content:    []anthropic.ContentBlock{{Type: "text", Text: "Hello from Claude"}},
			Usage:      anthropic.Usage{InputTokens: 12, OutputTokens: 8},
			StopReason: "end_turn",
		})
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})
	r.OnResponse(mr.capture())

	body := `{"model":"claude-3","messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 12, 8, 20)
	mr.assertOutputContent(t, "Hello from Claude")
}

func TestMock_Anthropic_to_OpenAI_Stream_OnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		msgStart, _ := json.Marshal(anthropic.Event{
			Type: "message_start",
			Message: &anthropic.MessageStart{
				ID: "msg-1", Type: "message", Role: "assistant", Model: "claude-3",
				Usage: anthropic.Usage{InputTokens: 10},
			},
		})
		fmt.Fprintf(w, "event: message_start\ndata: %s\n\n", msgStart)

		deltaData, _ := json.Marshal(anthropic.Delta{Type: "text_delta", Text: "Hello"})
		cbDelta, _ := json.Marshal(anthropic.Event{Type: "content_block_delta", Index: new(int), Delta: deltaData})
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", cbDelta)

		msgDeltaData, _ := json.Marshal(anthropic.Delta{StopReason: "end_turn"})
		msgDelta, _ := json.Marshal(anthropic.Event{Type: "message_delta", Delta: msgDeltaData, Usage: &anthropic.Usage{OutputTokens: 5}})
		fmt.Fprintf(w, "event: message_delta\ndata: %s\n\n", msgDelta)

		msgStop, _ := json.Marshal(anthropic.Event{Type: "message_stop"})
		fmt.Fprintf(w, "event: message_stop\ndata: %s\n\n", msgStop)
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})
	r.OnResponse(mr.capture())

	body := `{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 10, 5, 15)
	mr.assertOutputContent(t, "Hello")
}

func TestMock_MultiMessageInput_OnResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "cmpl-1", Object: "chat.completion", Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{{
				Index: 0, Message: &openai.ChatCompletionMessage{Role: "assistant", Content: "Response"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 50, CompletionTokens: 20, TotalTokens: 70},
		})
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})
	r.OnResponse(mr.capture())

	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "system", "content": "You are helpful"},
			{"role": "user", "content": "First question"},
			{"role": "assistant", "content": "First answer"},
			{"role": "user", "content": "Second question"}
		]
	}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 50, 20, 70)

	mr.mu.Lock()
	input := mr.input.(*openai.ChatCompletionRequest)
	if len(input.Messages) != 4 {
		t.Fatalf("expected 4 input messages, got %d", len(input.Messages))
	}
	if input.Messages[0].Role != "system" || input.Messages[0].Content != "You are helpful" {
		t.Errorf("system message = %+v", input.Messages[0])
	}
	if input.Messages[3].Content != "Second question" {
		t.Errorf("last user message = %q, want 'Second question'", input.Messages[3].Content)
	}
	mr.mu.Unlock()
}

func TestMock_OnResponse_RequestID(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "cmpl-1", Object: "chat.completion", Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{{
				Index: 0, Message: &openai.ChatCompletionMessage{Role: "assistant", Content: "hi"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		})
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	mr.assertCalled(t)
	mr.mu.Lock()
	if mr.requestID == "" {
		t.Error("expected non-empty requestID")
	}
	if !strings.HasPrefix(mr.requestID, "req_") {
		t.Errorf("requestID = %q, want prefix 'req_'", mr.requestID)
	}
	mr.mu.Unlock()
}

func TestMock_OnResponse_Duration(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "cmpl-1", Object: "chat.completion", Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{{
				Index: 0, Message: &openai.ChatCompletionMessage{Role: "assistant", Content: "hi"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		})
	}))
	defer upstream.Close()

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	mr.assertCalled(t)
	mr.mu.Lock()
	if mr.output == nil {
		t.Fatal("expected output")
	}
	mr.mu.Unlock()
}

// ============================================================================
// Real API Tests (gated by environment variables)
// ============================================================================

func skipIfNoRealAPI(t *testing.T) {
	if os.Getenv("TEST_REAL_API") != "1" {
		t.Skip("set TEST_REAL_API=1 to run real API tests")
	}
}

func TestReal_OpenAI_ChatCompletion(t *testing.T) {
	skipIfNoRealAPI(t)

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY not set")
	}

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{
		Name:    "openai",
		Type:    providers.ProviderTypeOpenAI,
		BaseURL: "https://api.openai.com",
		APIKey:  apiKey,
		Models:  []string{"gpt-4o-mini"},
	})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say hello in one word"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)

	mr.mu.Lock()
	if mr.output.Usage.TotalTokens == 0 {
		t.Error("expected non-zero usage")
	}
	if mr.output.Choices[0].Message.Content == "" {
		t.Error("expected non-empty response content")
	}
	mr.mu.Unlock()
}

func TestReal_OpenAI_Stream(t *testing.T) {
	skipIfNoRealAPI(t)

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY not set")
	}

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{
		Name:    "openai",
		Type:    providers.ProviderTypeOpenAI,
		BaseURL: "https://api.openai.com",
		APIKey:  apiKey,
		Models:  []string{"gpt-4o-mini"},
	})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Count 1 to 3"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)

	mr.mu.Lock()
	if mr.output.Usage.TotalTokens == 0 {
		t.Error("expected non-zero usage from stream")
	}
	mr.mu.Unlock()
}

func TestReal_Anthropic_ChatCompletion(t *testing.T) {
	skipIfNoRealAPI(t)

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Fatal("ANTHROPIC_API_KEY not set")
	}

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{
		Name:             "anthropic",
		Type:             providers.ProviderTypeAnthropic,
		BaseURL:          "https://api.anthropic.com",
		APIKey:           apiKey,
		Models:           []string{"claude-3-5-haiku-latest"},
		DefaultMaxTokens: 100,
	})
	r.OnResponse(mr.capture())

	body := `{"model":"claude-3-5-haiku-latest","messages":[{"role":"user","content":"Say hello in one word"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)

	mr.mu.Lock()
	if mr.output.Usage.TotalTokens == 0 {
		t.Error("expected non-zero usage")
	}
	if mr.output.Choices[0].Message.Content == "" {
		t.Error("expected non-empty response content")
	}
	mr.mu.Unlock()
}

func TestReal_Anthropic_Stream(t *testing.T) {
	skipIfNoRealAPI(t)

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Fatal("ANTHROPIC_API_KEY not set")
	}

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{
		Name:             "anthropic",
		Type:             providers.ProviderTypeAnthropic,
		BaseURL:          "https://api.anthropic.com",
		APIKey:           apiKey,
		Models:           []string{"claude-3-5-haiku-latest"},
		DefaultMaxTokens: 100,
	})
	r.OnResponse(mr.capture())

	body := `{"model":"claude-3-5-haiku-latest","messages":[{"role":"user","content":"Count 1 to 3"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)

	mr.mu.Lock()
	if mr.output.Usage.TotalTokens == 0 {
		t.Error("expected non-zero usage from stream")
	}
	mr.mu.Unlock()
}

func TestReal_OpenAI_to_Anthropic_Stream(t *testing.T) {
	skipIfNoRealAPI(t)

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY not set")
	}

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{
		Name:    "openai",
		Type:    providers.ProviderTypeOpenAI,
		BaseURL: "https://api.openai.com",
		APIKey:  apiKey,
		Models:  []string{"gpt-4o-mini"},
	})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4o-mini","max_tokens":100,"messages":[{"role":"user","content":"Count 1 to 3"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	bodyStr := w.Body.String()
	if !strings.Contains(bodyStr, "message_start") {
		t.Errorf("expected message_start in SSE output, got: %s", bodyStr)
	}

	mr.assertCalled(t)
	mr.assertNoError(t)

	mr.mu.Lock()
	if mr.output.Usage.TotalTokens == 0 {
		t.Error("expected non-zero usage")
	}
	mr.mu.Unlock()
}

func TestReal_Anthropic_to_OpenAI_Stream(t *testing.T) {
	skipIfNoRealAPI(t)

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Fatal("ANTHROPIC_API_KEY not set")
	}

	mr := &mockResponse{}
	r := NewProxy()
	r.AddProvider(&providers.Provider{
		Name:             "anthropic",
		Type:             providers.ProviderTypeAnthropic,
		BaseURL:          "https://api.anthropic.com",
		APIKey:           apiKey,
		Models:           []string{"claude-3-5-haiku-latest"},
		DefaultMaxTokens: 100,
	})
	r.OnResponse(mr.capture())

	body := `{"model":"claude-3-5-haiku-latest","messages":[{"role":"user","content":"Count 1 to 3"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	mr.assertCalled(t)
	mr.assertNoError(t)

	mr.mu.Lock()
	if mr.output.Usage.TotalTokens == 0 {
		t.Error("expected non-zero usage")
	}
	if mr.output.Choices[0].Message.Content == "" {
		t.Error("expected non-empty response content")
	}
	mr.mu.Unlock()
}
