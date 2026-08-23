package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNewClient(t *testing.T) {
	c, err := NewClient(&Configuration{API: "https://api.example.com", APIKey: "test-key"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.config.API != "https://api.example.com" {
		t.Errorf("API = %q, want https://api.example.com", c.config.API)
	}
	if c.config.APIKey != "test-key" {
		t.Errorf("APIKey = %q, want test-key", c.config.APIKey)
	}
}

func TestSetHTTPClient(t *testing.T) {
	c, _ := NewClient(&Configuration{API: "https://api.example.com", APIKey: "test-key"})
	customClient := &http.Client{Timeout: 10 * time.Second}
	c.SetHTTPClient(customClient)
	if c.client != customClient {
		t.Error("HTTP client was not set")
	}
}

func TestRawRequestPreservesResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("path = %q, want /v1/responses", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q", got)
		}
		body, _ := io.ReadAll(r.Body)
		if got := string(body); got != `{"unknown_extension":true}` {
			t.Errorf("body = %s", got)
		}
		w.Header().Set("X-Upstream", "preserved")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"custom_error":"slow_down"}`))
	}))
	defer server.Close()

	client, _ := NewClient(&Configuration{API: server.URL + "/v1", APIKey: "test-key"})
	request, err := client.NewRequest(context.Background(), http.MethodPost, "responses", strings.NewReader(`{"unknown_extension":true}`))
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusTooManyRequests || response.Header.Get("X-Upstream") != "preserved" || string(body) != `{"custom_error":"slow_down"}` {
		t.Fatalf("raw response = status %d, headers %v, body %s", response.StatusCode, response.Header, body)
	}
}

func TestModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", auth)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"data": []Model{
				{ID: "gpt-4", Object: "model", OwnedBy: "openai"},
				{ID: "gpt-3.5-turbo", Object: "model", OwnedBy: "openai"},
			},
		})
	}))
	defer server.Close()

	c, _ := NewClient(&Configuration{API: server.URL, APIKey: "test-key"})
	models, err := c.Models()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].ID != "gpt-4" {
		t.Errorf("models[0].ID = %q, want gpt-4", models[0].ID)
	}
}

func TestModels_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	c, _ := NewClient(&Configuration{API: server.URL, APIKey: "test-key"})
	_, err := c.Models()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestCreateChatCompletion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		json.NewEncoder(w).Encode(ChatCompletionResponse{
			ID:      "cmpl-123",
			Object:  "chat.completion",
			Model:   "gpt-4",
			Created: 1234567890,
			Choices: []ChatCompletionChoice{{
				Index:        0,
				Message:      &ChatCompletionMessage{Role: "assistant", Content: "Hello!"},
				FinishReason: "stop",
			}},
			Usage: &CompletionUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
		})
	}))
	defer server.Close()

	c, _ := NewClient(&Configuration{API: server.URL, APIKey: "test-key"})
	resp, err := c.CreateChatCompletion(context.Background(), &ChatCompletionRequest{
		Model:    "gpt-4",
		Messages: []ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ID != "cmpl-123" {
		t.Errorf("ID = %q, want cmpl-123", resp.ID)
	}
	if resp.Choices[0].Message.Content != "Hello!" {
		t.Errorf("content = %q, want Hello!", resp.Choices[0].Message.Content)
	}
	if resp.Usage.TotalTokens != 8 {
		t.Errorf("total_tokens = %d, want 8", resp.Usage.TotalTokens)
	}
}

func TestCreateChatCompletion_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]string{"message": "invalid key"},
		})
	}))
	defer server.Close()

	c, _ := NewClient(&Configuration{API: server.URL, APIKey: "bad-key"})
	_, err := c.CreateChatCompletion(context.Background(), &ChatCompletionRequest{
		Model:    "gpt-4",
		Messages: []ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestCreateChatCompletionStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []ChatCompletionResponse{
			{ID: "cmpl-1", Object: "chat.completion.chunk", Choices: []ChatCompletionChoice{{Index: 0, Delta: &ChatCompletionMessage{Role: "assistant", Content: "Hel"}}}},
			{ID: "cmpl-1", Object: "chat.completion.chunk", Choices: []ChatCompletionChoice{{Index: 0, Delta: &ChatCompletionMessage{Content: "lo"}}}},
			{ID: "cmpl-1", Object: "chat.completion.chunk", Choices: []ChatCompletionChoice{{Index: 0, Delta: &ChatCompletionMessage{}, FinishReason: "stop"}}, Usage: &CompletionUsage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}},
		}
		for _, c := range chunks {
			data, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	c, _ := NewClient(&Configuration{API: server.URL, APIKey: "test-key"})
	ch, err := c.CreateChatCompletionStream(context.Background(), &ChatCompletionRequest{
		Model:    "gpt-4",
		Messages: []ChatCompletionMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var contents []string
	for chunk := range ch {
		if chunk.Choices[0].Delta != nil && chunk.Choices[0].Delta.Content != "" {
			contents = append(contents, chunk.Choices[0].Delta.Content)
		}
	}
	if len(contents) != 2 {
		t.Fatalf("expected 2 content chunks, got %d", len(contents))
	}
	if contents[0] != "Hel" || contents[1] != "lo" {
		t.Errorf("contents = %v, want [Hel lo]", contents)
	}
}

func TestCreateChatCompletionStream_ContextCancel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		data, _ := json.Marshal(ChatCompletionResponse{
			ID: "cmpl-1", Object: "chat.completion.chunk",
			Choices: []ChatCompletionChoice{{Index: 0, Delta: &ChatCompletionMessage{Content: "test"}}},
		})
		fmt.Fprintf(w, "data: %s\n\n", data)
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	c, _ := NewClient(&Configuration{API: server.URL, APIKey: "test-key"})
	ch, err := c.CreateChatCompletionStream(ctx, &ChatCompletionRequest{
		Model:    "gpt-4",
		Messages: []ChatCompletionMessage{{Role: "user", Content: "hi"}},
		Stream:   true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	<-ch
	cancel()

	for range ch {
	}
}

func TestCreateChatCompletionStream_ReportsPrematureClose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		data, _ := json.Marshal(ChatCompletionResponse{
			Choices: []ChatCompletionChoice{{
				Index: 0,
				Delta: &ChatCompletionMessage{Role: RoleAssistant, Content: "partial"},
			}},
		})
		fmt.Fprintf(w, "data: %s\n\n", data)
	}))
	defer server.Close()

	c, _ := NewClient(&Configuration{API: server.URL, APIKey: "test-key"})
	ch, err := c.CreateChatCompletionStream(context.Background(), &ChatCompletionRequest{
		Model: "gpt-4", Stream: true,
	})
	if err != nil {
		t.Fatalf("CreateChatCompletionStream: %v", err)
	}

	var streamErr *Error
	for chunk := range ch {
		if chunk.Error != nil {
			streamErr = chunk.Error
		}
	}
	if streamErr == nil || streamErr.Type != "stream_error" {
		t.Fatalf("stream error = %#v, want stream_error", streamErr)
	}
}

func TestCreateChatCompletionStream_ReportsInvalidEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: not-json\n\n")
	}))
	defer server.Close()

	c, _ := NewClient(&Configuration{API: server.URL, APIKey: "test-key"})
	ch, err := c.CreateChatCompletionStream(context.Background(), &ChatCompletionRequest{
		Model: "gpt-4", Stream: true,
	})
	if err != nil {
		t.Fatalf("CreateChatCompletionStream: %v", err)
	}

	chunk := <-ch
	if chunk.Error == nil || !strings.Contains(chunk.Error.Message, "decode stream event") {
		t.Fatalf("stream error = %#v", chunk.Error)
	}
}

func TestCreateEmbeddings(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %q, want /embeddings", r.URL.Path)
		}
		json.NewEncoder(w).Encode(EmbeddingResponse{
			Object: "list",
			Data:   []Embedding{{Object: "embedding", Index: 0, Embedding: []float64{0.1, 0.2, 0.3}}},
			Model:  "text-embedding-ada-002",
			Usage:  &Usage{PromptTokens: 5, TotalTokens: 5},
		})
	}))
	defer server.Close()

	c, _ := NewClient(&Configuration{API: server.URL, APIKey: "test-key"})
	resp, err := c.CreateEmbeddings(context.Background(), &EmbeddingRequest{
		Input: "hello world",
		Model: "text-embedding-ada-002",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Data))
	}
	if len(resp.Data[0].Embedding) != 3 {
		t.Errorf("embedding length = %d, want 3", len(resp.Data[0].Embedding))
	}
}

func TestChatCompletionResponse_GetFirstChoice(t *testing.T) {
	resp := &ChatCompletionResponse{
		Choices: []ChatCompletionChoice{
			{Index: 0, Message: &ChatCompletionMessage{Content: "first"}},
			{Index: 1, Message: &ChatCompletionMessage{Content: "second"}},
		},
	}
	choice := resp.GetFirstChoice()
	if choice == nil {
		t.Fatal("expected choice, got nil")
	}
	if choice.Message.Content != "first" {
		t.Errorf("content = %q, want first", choice.Message.Content)
	}
}

func TestChatCompletionResponse_GetFirstChoice_Empty(t *testing.T) {
	resp := &ChatCompletionResponse{}
	choice := resp.GetFirstChoice()
	if choice != nil {
		t.Errorf("expected nil, got %+v", choice)
	}
}

func TestChatCompletionResponse_GetMessage_FromDelta(t *testing.T) {
	resp := &ChatCompletionResponse{
		Choices: []ChatCompletionChoice{{Index: 0, Delta: &ChatCompletionMessage{Role: "assistant", Content: "streaming"}}},
	}
	msg := resp.GetMessage()
	if msg == nil {
		t.Fatal("expected message, got nil")
	}
	if msg.Content != "streaming" {
		t.Errorf("content = %q, want streaming", msg.Content)
	}
}

func TestChatCompletionResponse_GetMessage_FromMessage(t *testing.T) {
	resp := &ChatCompletionResponse{
		Choices: []ChatCompletionChoice{{Index: 0, Message: &ChatCompletionMessage{Role: "assistant", Content: "complete"}}},
	}
	msg := resp.GetMessage()
	if msg == nil {
		t.Fatal("expected message, got nil")
	}
	if msg.Content != "complete" {
		t.Errorf("content = %q, want complete", msg.Content)
	}
}

func TestChatCompletionResponse_GetMessage_Empty(t *testing.T) {
	resp := &ChatCompletionResponse{}
	msg := resp.GetMessage()
	if msg != nil {
		t.Errorf("expected nil, got %+v", msg)
	}
}

func TestChatCompletionMessage_IsEmpty(t *testing.T) {
	if !(&ChatCompletionMessage{}).IsEmpty() {
		t.Error("empty message should be empty")
	}
	if (&ChatCompletionMessage{Content: "hi"}).IsEmpty() {
		t.Error("message with content should not be empty")
	}
	if (&ChatCompletionMessage{Role: "user"}).IsEmpty() {
		t.Error("message with role should not be empty")
	}
	if (&ChatCompletionMessage{ToolCalls: []ToolCall{{}}}).IsEmpty() {
		t.Error("message with tool calls should not be empty")
	}
}

func TestChatCompletionMessage_HasToolCall(t *testing.T) {
	if (&ChatCompletionMessage{}).HasToolCall() {
		t.Error("empty message should not have tool call")
	}
	if !(&ChatCompletionMessage{ToolCalls: []ToolCall{{ID: "tc-1"}}}).HasToolCall() {
		t.Error("message with tool calls should have tool call")
	}
}

func TestMessageHelpers(t *testing.T) {
	um := UserMessage("hello")
	if um.Role != "user" || um.Content != "hello" {
		t.Errorf("UserMessage = %+v", um)
	}

	sm := SystemMessage("be helpful")
	if sm.Role != "system" || sm.Content != "be helpful" {
		t.Errorf("SystemMessage = %+v", sm)
	}

	am := AssistantMessage("sure")
	if am.Role != "assistant" || am.Content != "sure" {
		t.Errorf("AssistantMessage = %+v", am)
	}

	tools := []ToolCall{{ID: "tc-1", Type: "function", Function: FunctionCall{Name: "get_weather"}}}
	atm := AssistantMessageWithTools("checking weather", tools)
	if atm.Role != "assistant" || len(atm.ToolCalls) != 1 {
		t.Errorf("AssistantMessageWithTools = %+v", atm)
	}

	trm := ToolResultMessage("tc-1", "get_weather", "sunny")
	if trm.Role != "tool" || trm.ToolCallID != "tc-1" || trm.Name != "get_weather" {
		t.Errorf("ToolResultMessage = %+v", trm)
	}
}

func TestChatCompletionMessage_UnmarshalJSON_StringContent(t *testing.T) {
	data := `{"role":"assistant","content":"hello"}`
	var msg ChatCompletionMessage
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg.Content != "hello" {
		t.Errorf("content = %q, want hello", msg.Content)
	}
}

func TestChatCompletionMessage_UnmarshalJSON_ArrayContent(t *testing.T) {
	data := `{"role":"user","content":[{"type":"text","text":"hello"},{"type":"text","text":"world"}]}`
	var msg ChatCompletionMessage
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg.Content != "hello\nworld" {
		t.Errorf("content = %q, want hello\\nworld", msg.Content)
	}
}

func TestChatCompletionMessage_UnmarshalJSON_EmptyContent(t *testing.T) {
	data := `{"role":"assistant"}`
	var msg ChatCompletionMessage
	if err := json.Unmarshal([]byte(data), &msg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg.Content != "" {
		t.Errorf("content = %q, want empty", msg.Content)
	}
}

func TestMakeRequest_InvalidJSON(t *testing.T) {
	c, _ := NewClient(&Configuration{API: "http://localhost", APIKey: "test"})
	_, err := c.MakeRequest(context.Background(), "/test", make(chan int))
	if err == nil {
		t.Fatal("expected error for unmarshalable data")
	}
	if !strings.Contains(err.Error(), "json error") {
		t.Errorf("error = %q, want json error", err.Error())
	}
}

func TestMakeRequest_ConnectionError(t *testing.T) {
	c, _ := NewClient(&Configuration{API: "http://127.0.0.1:1", APIKey: "test"})
	_, err := c.MakeRequest(context.Background(), "/test", nil)
	if err == nil {
		t.Fatal("expected error for connection failure")
	}
}
