package codec

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAnthropicToResponsesRequestPreservesToolLoop(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.6-sol","max_tokens":4096,"stream":true,
		"system":[{"type":"text","text":"Use tools precisely.","cache_control":{"type":"ephemeral"}}],
		"messages":[
			{"role":"user","content":[{"type":"text","text":"Read the file"}]},
			{"role":"assistant","content":[{"type":"thinking","thinking":"I should inspect it","signature":"signed"},{"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"proof.txt"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"hello"}]}
		],
		"tools":[{"name":"read_file","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
		"tool_choice":{"type":"auto","disable_parallel_tool_use":true},
		"metadata":{"user_id":"claude-code"},"thinking":{"type":"enabled","budget_tokens":1024},
		"context_management":{"edits":[]},"output_config":{"effort":"high"}
	}`)

	translated, request, diagnostics, err := TranslateRequest(Anthropic, OpenAIResponses, body)
	if err != nil {
		t.Fatal(err)
	}
	if len(BlockingDiagnostics(diagnostics)) > 0 {
		t.Fatalf("blocking diagnostics: %+v", diagnostics)
	}
	if !hasDiagnostic(diagnostics, "$.thinking", "warning") || !hasDiagnostic(diagnostics, "$.system[0].cache_control", "warning") {
		t.Fatalf("expected provider-specific warnings, got %+v", diagnostics)
	}
	if !request.Stream || len(request.Tools) != 1 || len(request.Messages) != 4 {
		t.Fatalf("canonical request lost semantics: %+v", request)
	}
	var got map[string]any
	if err := json.Unmarshal(translated, &got); err != nil {
		t.Fatal(err)
	}
	input := got["input"].([]any)
	assertItemType(t, input, "function_call")
	assertItemType(t, input, "function_call_output")
	tools := got["tools"].([]any)
	if tools[0].(map[string]any)["name"] != "read_file" {
		t.Fatalf("tool definition not preserved: %#v", tools)
	}
}

func TestResponsesRequestRejectsUnknownSemanticsAndCustomTools(t *testing.T) {
	tests := []struct {
		name string
		body string
		path string
		text string
	}{
		{
			name: "unknown top-level field",
			body: `{"model":"gpt-5-codex","input":"hi","additional_tools":[]}`,
			path: "$.additional_tools",
			text: "field is not supported",
		},
		{
			name: "stateful response continuation",
			body: `{"model":"gpt-5-codex","input":"hi","previous_response_id":"resp_1"}`,
			path: "$.previous_response_id",
			text: "field is not supported",
		},
		{
			name: "custom tool",
			body: `{"model":"gpt-5-codex","input":"hi","tools":[{"type":"computer","display_width":1024,"display_height":768}]}`,
			path: "$.tools[0].type",
			text: `tool type "computer" is not portable`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, diagnostics, err := TranslateRequest(OpenAIResponses, Anthropic, []byte(test.body))
			if err == nil {
				t.Fatal("expected translation error")
			}
			if !hasDiagnostic(diagnostics, test.path, "") || !strings.Contains(err.Error(), test.text) {
				t.Fatalf("diagnostics = %+v, error = %v", diagnostics, err)
			}
		})
	}
}

func TestResponsesToAnthropicResponsePreservesToolCall(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","object":"response","created_at":1700000000,"status":"completed","model":"gpt-5.6-sol",
		"output":[
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Need the file"}]},
			{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Checking.","annotations":[]}]},
			{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"proof.txt\"}"}
		],
		"usage":{"input_tokens":20,"output_tokens":8,"total_tokens":28,"input_tokens_details":{"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":4}}
	}`)

	translated, response, diagnostics, err := TranslateResponse(OpenAIResponses, Anthropic, body)
	if err != nil {
		t.Fatal(err)
	}
	if len(BlockingDiagnostics(diagnostics)) > 0 {
		t.Fatalf("blocking diagnostics: %+v", diagnostics)
	}
	if response.StopReason != "tool_use" || len(response.Content) != 3 {
		t.Fatalf("canonical response lost semantics: %+v", response)
	}
	var got map[string]any
	if err := json.Unmarshal(translated, &got); err != nil {
		t.Fatal(err)
	}
	if got["stop_reason"] != "tool_use" {
		t.Fatalf("stop reason = %#v", got["stop_reason"])
	}
	content := got["content"].([]any)
	assertItemType(t, content, "thinking")
	assertItemType(t, content, "text")
	assertItemType(t, content, "tool_use")
}

func TestResponsesRefusalIsNotRenderedAsOrdinaryText(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","object":"response","status":"completed","model":"model",
		"output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"cannot comply"}]}]
	}`)
	translated, _, diagnostics, err := TranslateResponse(OpenAIResponses, Anthropic, body)
	if err == nil {
		t.Fatalf("expected refusal conversion to fail, got %s", translated)
	}
	if !hasDiagnostic(diagnostics, "$.output[0].content[0].type", "") || !strings.Contains(err.Error(), "refusal content") {
		t.Fatalf("diagnostics = %+v, error = %v", diagnostics, err)
	}
}

func TestResponsesStreamToAnthropicToolUse(t *testing.T) {
	input := strings.Join([]string{
		sseEvent("response.created", `{"type":"response.created","response":{"id":"resp_1","status":"in_progress","model":"gpt-5.6-sol"}}`),
		sseEvent("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file","arguments":""}}`),
		sseEvent("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"path\":"}`),
		sseEvent("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"\"proof.txt\"}"}`),
		sseEvent("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"proof.txt\"}"}}`),
		sseEvent("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-5.6-sol","output":[],"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15,"input_tokens_details":{"cached_tokens":2},"output_tokens_details":{"reasoning_tokens":0}}}}`),
	}, "")
	recorder := httptest.NewRecorder()
	response, err := TranslateStream(context.Background(), OpenAIResponses, Anthropic, strings.NewReader(input), recorder)
	if err != nil {
		t.Fatal(err)
	}
	output := recorder.Body.String()
	for _, want := range []string{"event: message_start", `"type":"tool_use"`, `"type":"input_json_delta"`, `"partial_json":"{\"path\":"`, `"stop_reason":"tool_use"`, "event: message_stop"} {
		if !strings.Contains(output, want) {
			t.Errorf("missing %q in:\n%s", want, output)
		}
	}
	if response.Usage == nil || response.Usage.TotalTokens != 15 || len(response.Content) != 1 || response.Content[0].Name != "read_file" {
		t.Fatalf("assembled response = %+v", response)
	}
}

func TestAnthropicStreamToResponsesToolUse(t *testing.T) {
	input := strings.Join([]string{
		sseEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus","content":[],"usage":{"input_tokens":9,"output_tokens":0}}}`),
		sseEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}`),
		sseEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"proof.txt\"}"}}`),
		sseEvent("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sseEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":4}}`),
		sseEvent("message_stop", `{"type":"message_stop"}`),
	}, "")
	recorder := httptest.NewRecorder()
	response, err := TranslateStream(context.Background(), Anthropic, OpenAIResponses, strings.NewReader(input), recorder)
	if err != nil {
		t.Fatal(err)
	}
	output := recorder.Body.String()
	for _, want := range []string{"event: response.created", `"type":"function_call"`, "event: response.function_call_arguments.delta", "event: response.function_call_arguments.done", "event: response.output_item.done", "event: response.completed"} {
		if !strings.Contains(output, want) {
			t.Errorf("missing %q in:\n%s", want, output)
		}
	}
	if response.StopReason != "tool_use" || response.Usage == nil || response.Usage.TotalTokens != 13 {
		t.Fatalf("assembled response = %+v", response)
	}
}

func TestWriteResponseStreamFallback(t *testing.T) {
	recorder := httptest.NewRecorder()
	response := &Response{
		ID: "resp_1", Model: "model", StopReason: "tool_use",
		Content: []Content{
			{Type: ContentText, Text: "checking"},
			{Type: ContentToolCall, ID: "call_1", Name: "read_file", Arguments: map[string]any{"path": "proof.txt"}},
		},
		Usage: &Usage{InputTokens: 4, OutputTokens: 2, TotalTokens: 6},
	}
	if err := WriteResponseStream(Anthropic, response, recorder); err != nil {
		t.Fatal(err)
	}
	output := recorder.Body.String()
	for _, want := range []string{`"type":"text_delta"`, `"type":"tool_use"`, `"type":"input_json_delta"`, `"stop_reason":"tool_use"`, "event: message_stop"} {
		if !strings.Contains(output, want) {
			t.Errorf("missing %q in:\n%s", want, output)
		}
	}
}

func sseEvent(event, data string) string {
	return "event: " + event + "\ndata: " + data + "\n\n"
}

func assertItemType(t *testing.T, items []any, typeName string) {
	t.Helper()
	for _, raw := range items {
		if item, ok := raw.(map[string]any); ok && item["type"] == typeName {
			return
		}
	}
	t.Fatalf("missing item type %q in %#v", typeName, items)
}

func hasDiagnostic(diagnostics []Diagnostic, path, severity string) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Path == path && diagnostic.Severity == severity {
			return true
		}
	}
	return false
}
