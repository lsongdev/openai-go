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

func TestResponses_OpenAI_NonStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify upstream got a chat completion request.
		var got openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&got)
		if len(got.Messages) < 2 || got.Messages[0].Role != "system" {
			t.Errorf("expected instructions as system message, got %+v", got.Messages)
		}
		if got.Messages[len(got.Messages)-1].Content != "Hello there" {
			t.Errorf("expected last user message 'Hello there', got %q", got.Messages[len(got.Messages)-1].Content)
		}
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "chatcmpl-1", Object: "chat.completion", Model: "gpt-4o", Created: 1700000000,
			Choices: []openai.ChatCompletionChoice{{
				Index:        0,
				Message:      &openai.ChatCompletionMessage{Role: "assistant", Content: "Hi!"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 7, CompletionTokens: 2, TotalTokens: 9},
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4o"}})

	body := `{"model":"gpt-4o","instructions":"Be helpful.","input":"Hello there"}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp openai.ResponseObject
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Object != "response" {
		t.Errorf("object = %q, want response", resp.Object)
	}
	if resp.Status != "completed" {
		t.Errorf("status = %q, want completed", resp.Status)
	}
	if len(resp.Output) != 1 || resp.Output[0].Type != "message" {
		t.Fatalf("expected one message output item, got %+v", resp.Output)
	}
	if len(resp.Output[0].Content) != 1 || resp.Output[0].Content[0].Text != "Hi!" {
		t.Errorf("output content = %+v, want Hi!", resp.Output[0].Content)
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 7 || resp.Usage.OutputTokens != 2 || resp.Usage.TotalTokens != 9 {
		t.Errorf("usage = %+v, want {7,2,9}", resp.Usage)
	}
	if resp.Instructions != "Be helpful." {
		t.Errorf("instructions = %q", resp.Instructions)
	}
}

func TestResponses_OpenAI_NonStream_InputArray(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&got)
		// Expect: system + user "first" + assistant "answer" + user "follow-up"
		if len(got.Messages) != 4 {
			t.Errorf("expected 4 messages, got %d (%+v)", len(got.Messages), got.Messages)
		}
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "chatcmpl-2", Object: "chat.completion", Model: "gpt-4o",
			Choices: []openai.ChatCompletionChoice{{
				Index:        0,
				Message:      &openai.ChatCompletionMessage{Role: "assistant", Content: "ok"},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4o"}})

	body := `{
		"model": "gpt-4o",
		"instructions": "sys",
		"input": [
			{"type": "message", "role": "user", "content": "first"},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "follow-up"}]}
		]
	}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestResponses_OpenAI_Stream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []openai.ChatCompletionResponse{
			{ID: "chatcmpl-1", Object: "chat.completion.chunk", Model: "gpt-4o", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Role: "assistant", Content: "Hel"}}}},
			{ID: "chatcmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{Content: "lo"}}}},
			{ID: "chatcmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{}, FinishReason: "stop"}}, Usage: &openai.CompletionUsage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}},
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
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4o"}})
	r.OnResponse(mr.capture())

	body := `{"model":"gpt-4o","input":"hi","stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	out := w.Body.String()
	for _, want := range []string{
		"event: response.created",
		"event: response.in_progress",
		"event: response.output_item.added",
		"event: response.content_part.added",
		"event: response.output_text.delta",
		"event: response.output_text.done",
		"event: response.content_part.done",
		"event: response.output_item.done",
		"event: response.completed",
		`"delta":"Hel"`,
		`"delta":"lo"`,
		`"text":"Hello"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in stream output", want)
		}
	}

	mr.assertCalled(t)
	mr.assertNoError(t)
	mr.assertUsage(t, 3, 2, 5)
	mr.assertOutputContent(t, "Hello")
}

func TestResponses_OpenAI_Stream_ToolCalls(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []openai.ChatCompletionResponse{
			{ID: "chatcmpl-1", Object: "chat.completion.chunk", Model: "gpt-4o", Choices: []openai.ChatCompletionChoice{{
				Index: 0,
				Delta: &openai.ChatCompletionMessage{Role: "assistant", ToolCalls: []openai.ToolCall{{
					Index: 0, ID: "call_abc", Type: "function",
					Function: openai.FunctionCall{Name: "get_weather", Arguments: ""},
				}}},
			}}},
			{ID: "chatcmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{
				Index: 0,
				Delta: &openai.ChatCompletionMessage{ToolCalls: []openai.ToolCall{{
					Index: 0, Function: openai.FunctionCall{Arguments: `{"city":`},
				}}},
			}}},
			{ID: "chatcmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{
				Index: 0,
				Delta: &openai.ChatCompletionMessage{ToolCalls: []openai.ToolCall{{
					Index: 0, Function: openai.FunctionCall{Arguments: `"SF"}`},
				}}},
			}}},
			{ID: "chatcmpl-1", Object: "chat.completion.chunk", Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &openai.ChatCompletionMessage{}, FinishReason: "tool_calls"}}, Usage: &openai.CompletionUsage{PromptTokens: 10, CompletionTokens: 3, TotalTokens: 13}},
		}
		for _, c := range chunks {
			data, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4o"}})

	body := `{
		"model": "gpt-4o",
		"input": "What's the weather?",
		"stream": true,
		"tools": [{"type":"function","name":"get_weather","description":"...","parameters":{"type":"object"}}]
	}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	out := w.Body.String()
	for _, want := range []string{
		"event: response.output_item.added",
		`"type":"function_call"`,
		"event: response.function_call_arguments.delta",
		`"delta":"{\"city\":"`,
		"event: response.function_call_arguments.done",
		`"arguments":"{\"city\":\"SF\"}"`,
		"event: response.completed",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in stream output\n--- full output ---\n%s", want, out)
		}
	}
}

func TestResponses_OpenAI_FunctionCallInput(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&got)
		// Expect: user, assistant (with tool_calls), tool result, user follow-up
		if len(got.Messages) != 4 {
			t.Fatalf("expected 4 messages, got %d: %+v", len(got.Messages), got.Messages)
		}
		assistant := got.Messages[1]
		if assistant.Role != "assistant" || len(assistant.ToolCalls) != 1 {
			t.Errorf("assistant message not preserved: %+v", assistant)
		}
		if assistant.ToolCalls[0].Function.Name != "get_weather" {
			t.Errorf("tool name = %q", assistant.ToolCalls[0].Function.Name)
		}
		toolResult := got.Messages[2]
		if toolResult.Role != "tool" || toolResult.Content != "sunny" {
			t.Errorf("tool result not preserved: %+v", toolResult)
		}
		json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "chatcmpl-3", Object: "chat.completion", Model: "gpt-4o",
			Choices: []openai.ChatCompletionChoice{{
				Index:        0,
				Message:      &openai.ChatCompletionMessage{Role: "assistant", Content: "It's sunny."},
				FinishReason: "stop",
			}},
			Usage: &openai.CompletionUsage{PromptTokens: 12, CompletionTokens: 4, TotalTokens: 16},
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: upstream.URL, Models: []string{"gpt-4o"}})

	body := `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "user", "content": "weather?"},
			{"type": "function_call", "id": "fc_1", "call_id": "call_abc", "name": "get_weather", "arguments": "{\"city\":\"SF\"}"},
			{"type": "function_call_output", "call_id": "call_abc", "output": "sunny"},
			{"type": "message", "role": "user", "content": "thanks"}
		]
	}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestResponses_Anthropic_NonStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(anthropic.Response{
			ID: "msg_x", Type: "message", Model: "claude-3",
			Content:    []anthropic.ContentBlock{{Type: "text", Text: "Hi from Claude"}},
			Usage:      anthropic.Usage{InputTokens: 11, OutputTokens: 4},
			StopReason: "end_turn",
		})
	}))
	defer upstream.Close()

	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "anth", Type: providers.ProviderTypeAnthropic, BaseURL: upstream.URL, Models: []string{"claude-3"}, DefaultMaxTokens: 4096})

	body := `{"model":"claude-3","instructions":"Be brief.","input":"hi"}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp openai.ResponseObject
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Output) != 1 || resp.Output[0].Content[0].Text != "Hi from Claude" {
		t.Errorf("unexpected output: %+v", resp.Output)
	}
	if resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 4 || resp.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestResponses_Anthropic_Stream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		msgStart, _ := json.Marshal(anthropic.Event{
			Type: "message_start",
			Message: &anthropic.MessageStart{
				ID: "msg_y", Type: "message", Role: "assistant", Model: "claude-3",
				Usage: anthropic.Usage{InputTokens: 8},
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

	body := `{"model":"claude-3","input":"hi","stream":true}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	out := w.Body.String()
	for _, want := range []string{
		"event: response.created",
		"event: response.output_text.delta",
		`"delta":"Hello"`,
		"event: response.completed",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q\nfull:\n%s", want, out)
		}
	}
}

func TestResponses_NoProvider(t *testing.T) {
	r := NewProxy()
	r.AddProvider(&providers.Provider{Name: "oai", Type: providers.ProviderTypeOpenAI, BaseURL: "http://localhost", Models: []string{"gpt-4o"}})

	body := `{"model":"unknown","input":"hi"}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}

func TestResponses_MethodNotAllowed(t *testing.T) {
	r := NewProxy()
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("GET", "/v1/responses", nil))
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}
