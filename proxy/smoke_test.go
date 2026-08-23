package proxy

import (
	"context"
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

func TestSmoke_Models(t *testing.T) {
	r := NewProxy()
	r.AddProvider(&providers.Provider{
		Name: "p1", Type: providers.ProviderTypeOpenAI, BaseURL: "http://localhost",
		Models: []string{"gpt-4", "gpt-3.5", "gpt-4"}, ModelAliases: map[string]string{"fast": "gpt-4"},
	})
	r.AddProvider(&providers.Provider{Name: "p2", Type: providers.ProviderTypeAnthropic, BaseURL: "http://localhost", Models: []string{"claude-3", "gpt-4"}})

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("GET", "/v1/models", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var body struct {
		Data []openai.Model `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if len(body.Data) != 4 {
		t.Fatalf("expected 4 unique public models, got %d: %#v", len(body.Data), body.Data)
	}
	owners := make(map[string]string, len(body.Data))
	for _, model := range body.Data {
		owners[model.ID] = model.OwnedBy
	}
	if owners["fast"] != "p1" || owners["gpt-4"] != "p1" {
		t.Fatalf("model aliases or deterministic ownership missing: %#v", owners)
	}
}

func TestSmoke_ChatCompletions_OpenAI(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "cmpl-test", Object: "chat.completion", Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{{
				Index: 0, Message: &openai.ChatCompletionMessage{Role: "assistant", Content: "Hello"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp openai.ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if resp.Choices[0].Message.Content != "Hello" {
		t.Errorf("content = %q, want Hello", resp.Choices[0].Message.Content)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("total_tokens = %d, want 15", resp.Usage.TotalTokens)
	}
}

func TestSmoke_ChatCompletions_OpenAI_Stream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []openai.ChatCompletionResponse{
			{ID: "cmpl-test", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Role: "assistant", Content: "Hel"}}}},
			{ID: "cmpl-test", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Content: "lo"}}}},
			{ID: "cmpl-test", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{}, FinishReason: "stop"}}, Usage: &openai.CompletionUsage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}},
		}
		for _, c := range chunks {
			data, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	bodyStr := w.Body.String()
	if !strings.Contains(bodyStr, "Hel") {
		t.Errorf("expected chunk with 'Hel', got: %s", bodyStr)
	}
	if !strings.Contains(bodyStr, "[DONE]") {
		t.Errorf("expected [DONE] in stream output, got: %s", bodyStr)
	}
}

func TestSmoke_ChatCompletions_Anthropic(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		json.NewEncoder(w).Encode(anthropic.Response{
			ID: "msg_abc", Type: "message", Model: "claude-3",
			Content: []anthropic.ContentBlock{
				{Type: "thinking", Thinking: "Let me think..."},
				{Type: "text", Text: "Hello world"},
			},
			Usage:      anthropic.Usage{InputTokens: 5, OutputTokens: 3},
			StopReason: "end_turn",
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})

	body := `{"model":"claude-3","messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp openai.ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if resp.Choices[0].Message.ReasoningContent != "Let me think..." {
		t.Errorf("reasoning = %q, want 'Let me think...'", resp.Choices[0].Message.ReasoningContent)
	}
	if resp.Choices[0].Message.Content != "Hello world" {
		t.Errorf("content = %q, want 'Hello world'", resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", resp.Choices[0].FinishReason)
	}
	if resp.Usage.TotalTokens != 8 {
		t.Errorf("total_tokens = %d, want 8", resp.Usage.TotalTokens)
	}
}

func TestSmoke_ChatCompletions_Anthropic_Stream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")

		msgStart, _ := json.Marshal(anthropic.Event{
			Type: "message_start",
			Message: &anthropic.MessageStart{
				ID: "msg_abc", Type: "message", Role: "assistant", Model: "claude-3",
				Usage: anthropic.Usage{InputTokens: 5},
			},
		})
		fmt.Fprintf(w, "event: message_start\ndata: %s\n\n", msgStart)

		deltaData, _ := json.Marshal(anthropic.Delta{Type: "text_delta", Text: "Hello"})
		cbDelta, _ := json.Marshal(anthropic.Event{Type: "content_block_delta", Index: new(int), Delta: deltaData})
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", cbDelta)

		msgDeltaData, _ := json.Marshal(anthropic.Delta{StopReason: "end_turn"})
		msgDelta, _ := json.Marshal(anthropic.Event{Type: "message_delta", Delta: msgDeltaData, Usage: &anthropic.Usage{OutputTokens: 3}})
		fmt.Fprintf(w, "event: message_delta\ndata: %s\n\n", msgDelta)

		msgStop, _ := json.Marshal(anthropic.Event{Type: "message_stop"})
		fmt.Fprintf(w, "event: message_stop\ndata: %s\n\n", msgStop)
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})

	body := `{"model":"claude-3","messages":[{"role":"user","content":"hi"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	bodyStr := w.Body.String()
	if !strings.Contains(bodyStr, "Hello") {
		t.Errorf("expected 'Hello' in stream output, got: %s", bodyStr)
	}
	if !strings.Contains(bodyStr, "[DONE]") {
		t.Errorf("expected [DONE] in stream output, got: %s", bodyStr)
	}
}

func TestSmoke_Messages_Anthropic(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		json.NewEncoder(w).Encode(anthropic.Response{
			ID: "msg_abc", Type: "message", Role: "assistant", Model: "claude-3",
			Content:    []anthropic.ContentBlock{{Type: "text", Text: "Hello from Anthropic"}},
			Usage:      anthropic.Usage{InputTokens: 3, OutputTokens: 2},
			StopReason: "end_turn",
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})

	body := `{"model":"claude-3","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp anthropic.Response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if resp.Content[0].Text != "Hello from Anthropic" {
		t.Errorf("text = %q, want 'Hello from Anthropic'", resp.Content[0].Text)
	}
}

func TestSmoke_Messages_OpenAI(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "cmpl-test", Object: "chat.completion", Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{{
				Message:      &openai.ChatCompletionMessage{Role: "assistant", Content: "Hello from OpenAI"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})

	body := `{"model":"gpt-4","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp anthropic.Response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if len(resp.Content) == 0 || resp.Content[0].Text != "Hello from OpenAI" {
		t.Errorf("content = %+v, want 'Hello from OpenAI'", resp.Content)
	}
	if resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 10 {
		t.Errorf("usage = %+v, want input=5 output=10", resp.Usage)
	}
}

func TestSmoke_Errors(t *testing.T) {
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "p", Type: providers.ProviderTypeOpenAI, BaseURL: "http://localhost", Models: []string{"m"}})

	t.Run("not found", func(t *testing.T) {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest("GET", "/v1/unknown", nil))
		if w.Code != http.StatusNotFound {
			t.Errorf("expected 404, got %d", w.Code)
		}
	})

	t.Run("method not allowed", func(t *testing.T) {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest("GET", "/v1/chat/completions", nil))
		if w.Code != http.StatusMethodNotAllowed {
			t.Errorf("expected 405, got %d", w.Code)
		}
	})

	t.Run("rejected by hook", func(t *testing.T) {
		r2 := NewProxy()
		r2.AddProvider(&providers.Provider{Name: "p", Type: providers.ProviderTypeOpenAI, BaseURL: "http://localhost", Models: []string{"m"}})
		r2.OnRequest(func(ctx *providers.RequestContext) error {
			return fmt.Errorf("blocked")
		})
		body := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
		w := httptest.NewRecorder()
		r2.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		if w.Code != http.StatusForbidden {
			t.Errorf("expected 403, got %d", w.Code)
		}
	})
}

func TestSmoke_OpenAIClient(t *testing.T) {
	// Mock upstream that responds to OpenAI chat completions
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "cmpl-test", Object: "chat.completion", Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{{
				Message:      &openai.ChatCompletionMessage{Role: "assistant", Content: "Hello from mock"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 7, CompletionTokens: 9, TotalTokens: 16},
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})

	// Start the router as an HTTP server
	routerServer := httptest.NewServer(r)
	defer routerServer.Close()

	// Use the openai client library to talk to the router
	client, err := openai.NewClient(&openai.Configuration{
		API:    routerServer.URL + "/v1",
		APIKey: "test-key",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	resp, err := client.CreateChatCompletion(context.Background(), &openai.ChatCompletionRequest{
		Model: "gpt-4",
		Messages: []openai.ChatCompletionMessage{
			{Role: "user", Content: "hi"},
		},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if resp.Choices[0].Message.Content != "Hello from mock" {
		t.Errorf("content = %q, want 'Hello from mock'", resp.Choices[0].Message.Content)
	}
	if resp.Usage.TotalTokens != 16 {
		t.Errorf("total_tokens = %d, want 16", resp.Usage.TotalTokens)
	}
}

func TestSmoke_OpenAIClient_Stream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []openai.ChatCompletionResponse{
			{ID: "cmpl-s", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Role: "assistant", Content: "Hel"}}}},
			{ID: "cmpl-s", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Content: "lo"}}}},
		}
		for _, c := range chunks {
			data, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})

	routerServer := httptest.NewServer(r)
	defer routerServer.Close()

	client, err := openai.NewClient(&openai.Configuration{
		API:    routerServer.URL + "/v1",
		APIKey: "test-key",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	ch, err := client.CreateChatCompletionStream(context.Background(), &openai.ChatCompletionRequest{
		Model: "gpt-4",
		Messages: []openai.ChatCompletionMessage{
			{Role: "user", Content: "hi"},
		},
		Stream: true,
	})
	if err != nil {
		t.Fatalf("CreateChatCompletionStream: %v", err)
	}

	var received []string
	for chunk := range ch {
		received = append(received, chunk.Choices[0].Delta.Content)
	}
	if len(received) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %v", len(received), received)
	}
	if received[0] != "Hel" || received[1] != "lo" {
		t.Errorf("chunks = %v, want [Hel lo]", received)
	}
}

func TestSmoke_AnthropicClient(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(anthropic.Response{
			ID: "msg_abc", Type: "message", Role: "assistant", Model: "claude-3",
			Content:    []anthropic.ContentBlock{{Type: "text", Text: "Hello from Anthropic"}},
			Usage:      anthropic.Usage{InputTokens: 4, OutputTokens: 6},
			StopReason: "end_turn",
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})

	routerServer := httptest.NewServer(r)
	defer routerServer.Close()

	// Use the anthropic client library to talk to the router at /v1/messages
	client := anthropic.NewClient(&anthropic.Configuration{
		API:    routerServer.URL,
		APIKey: "test-key",
	})

	resp, err := client.CreateMessage(context.Background(), &anthropic.Request{
		Model:     "claude-3",
		MaxTokens: 100,
		Messages:  []anthropic.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	if resp.Content[0].Text != "Hello from Anthropic" {
		t.Errorf("text = %q, want 'Hello from Anthropic'", resp.Content[0].Text)
	}
	if resp.Usage.InputTokens != 4 || resp.Usage.OutputTokens != 6 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestSmoke_AnthropicClient_Stream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		msgStart, _ := json.Marshal(anthropic.Event{
			Type: "message_start",
			Message: &anthropic.MessageStart{
				ID: "msg_abc", Type: "message", Role: "assistant", Model: "claude-3",
				Usage: anthropic.Usage{InputTokens: 3},
			},
		})
		fmt.Fprintf(w, "event: message_start\ndata: %s\n\n", msgStart)

		deltaData, _ := json.Marshal(anthropic.Delta{Type: "text_delta", Text: "Hel"})
		cbDelta, _ := json.Marshal(anthropic.Event{Type: "content_block_delta", Index: new(int), Delta: deltaData})
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", cbDelta)

		deltaData2, _ := json.Marshal(anthropic.Delta{Type: "text_delta", Text: "lo"})
		cbDelta2, _ := json.Marshal(anthropic.Event{Type: "content_block_delta", Index: new(int), Delta: deltaData2})
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", cbDelta2)

		msgDeltaData, _ := json.Marshal(anthropic.Delta{StopReason: "end_turn"})
		msgDelta, _ := json.Marshal(anthropic.Event{Type: "message_delta", Delta: msgDeltaData, Usage: &anthropic.Usage{OutputTokens: 2}})
		fmt.Fprintf(w, "event: message_delta\ndata: %s\n\n", msgDelta)

		msgStop, _ := json.Marshal(anthropic.Event{Type: "message_stop"})
		fmt.Fprintf(w, "event: message_stop\ndata: %s\n\n", msgStop)
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})

	routerServer := httptest.NewServer(r)
	defer routerServer.Close()

	client := anthropic.NewClient(&anthropic.Configuration{
		API:    routerServer.URL,
		APIKey: "test-key",
	})

	ms, err := client.CreateMessageStream(context.Background(), &anthropic.Request{
		Model:     "claude-3",
		MaxTokens: 100,
		Messages:  []anthropic.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}

	var received []string
	for event := range ms.Events {
		if event.Type == "content_block_delta" {
			var delta anthropic.Delta
			if err := json.Unmarshal(event.Delta, &delta); err != nil {
				continue
			}
			received = append(received, delta.Text)
		}
	}
	if len(received) != 2 {
		t.Fatalf("expected 2 deltas, got %d: %v", len(received), received)
	}
	if received[0] != "Hel" || received[1] != "lo" {
		t.Errorf("deltas = %v, want [Hel lo]", received)
	}
}

func TestSmoke_Messages_Anthropic_Stream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")

		msgStart, _ := json.Marshal(anthropic.Event{
			Type: "message_start",
			Message: &anthropic.MessageStart{
				ID: "msg_abc", Type: "message", Role: "assistant", Model: "claude-3",
				Usage: anthropic.Usage{InputTokens: 3},
			},
		})
		fmt.Fprintf(w, "event: message_start\ndata: %s\n\n", msgStart)

		deltaData, _ := json.Marshal(anthropic.Delta{Type: "text_delta", Text: "Hi"})
		cbDelta, _ := json.Marshal(anthropic.Event{Type: "content_block_delta", Index: new(int), Delta: deltaData})
		fmt.Fprintf(w, "event: content_block_delta\ndata: %s\n\n", cbDelta)

		msgDeltaData, _ := json.Marshal(anthropic.Delta{StopReason: "end_turn"})
		msgDelta, _ := json.Marshal(anthropic.Event{Type: "message_delta", Delta: msgDeltaData, Usage: &anthropic.Usage{OutputTokens: 2}})
		fmt.Fprintf(w, "event: message_delta\ndata: %s\n\n", msgDelta)

		msgStop, _ := json.Marshal(anthropic.Event{Type: "message_stop"})
		fmt.Fprintf(w, "event: message_stop\ndata: %s\n\n", msgStop)
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})

	body := `{"model":"claude-3","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	bodyStr := w.Body.String()
	if !strings.Contains(bodyStr, "event:") {
		t.Errorf("expected SSE events, got: %s", bodyStr)
	}
}

func TestSmoke_Messages_OpenAI_Stream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")

		chunks := []openai.ChatCompletionResponse{
			{ID: "cmpl-test", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Role: "assistant", Content: "Hi"}}}},
			{ID: "cmpl-test", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{}, FinishReason: "stop"}}, Usage: &openai.CompletionUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}},
		}
		for _, c := range chunks {
			data, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})

	body := `{"model":"gpt-4","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	bodyStr := w.Body.String()
	if !strings.Contains(bodyStr, "Hi") {
		t.Errorf("expected 'Hi' in stream output, got: %s", bodyStr)
	}
	if !strings.Contains(bodyStr, "message_stop") {
		t.Errorf("expected message_stop in stream output, got: %s", bodyStr)
	}
	if !strings.Contains(bodyStr, "content_block_delta") {
		t.Errorf("expected content_block_delta in stream output, got: %s", bodyStr)
	}
}

func TestSmoke_ArrayContent(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "cmpl-test", Object: "chat.completion", Model: "gpt-4",
			Choices: []openai.ChatCompletionChoice{{
				Message:      &openai.ChatCompletionMessage{Role: "assistant", Content: "Hello"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4"}})

	// content as array of parts
	body := `{"model":"gpt-4","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp openai.ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if resp.Choices[0].Message.Content != "Hello" {
		t.Errorf("content = %q, want 'Hello'", resp.Choices[0].Message.Content)
	}
}

func TestSmoke_ArrayContent_Messages(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(anthropic.Response{
			ID: "msg_abc", Type: "message", Role: "assistant", Model: "claude-3",
			Content:    []anthropic.ContentBlock{{Type: "text", Text: "Hello"}},
			Usage:      anthropic.Usage{InputTokens: 1, OutputTokens: 1},
			StopReason: "end_turn",
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})

	// content as array of blocks (Anthropic format)
	body := `{"model":"claude-3","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp anthropic.Response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if resp.Content[0].Text != "Hello" {
		t.Errorf("text = %q, want 'Hello'", resp.Content[0].Text)
	}
}
