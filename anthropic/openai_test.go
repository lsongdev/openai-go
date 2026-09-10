package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/lsongdev/miya-agents/openai"
)

func TestChatCompletionRequestPreservesTools(t *testing.T) {
	req := &openai.ChatCompletionRequest{
		Model: "claude-test",
		Tools: []openai.ToolDef{{
			Type: "function",
			Function: openai.FunctionDef{
				Name: "read_file", Description: "read a file",
				Parameters: map[string]any{"type": "object"},
			},
		}},
		Messages: []openai.ChatCompletionMessage{
			openai.UserMessage("read README.md"),
			{
				Role: openai.RoleAssistant,
				ToolCalls: []openai.ToolCall{{
					ID: "call_1", Type: "function",
					Function: openai.FunctionCall{Name: "read_file", Arguments: `{"path":"README.md"}`},
				}},
			},
			openai.ToolResultMessage("call_1", "read_file", "hello"),
		},
	}

	got := NewAnthropicRequestFromChatCompletionRequest(req)
	if len(got.Tools) != 1 || got.Tools[0].Name != "read_file" {
		t.Fatalf("tools = %#v", got.Tools)
	}
	if len(got.Messages) != 3 {
		t.Fatalf("messages = %#v", got.Messages)
	}
	use := got.Messages[1].Blocks[0]
	if use.Type != "tool_use" || use.ID != "call_1" || use.Name != "read_file" || string(use.Input) != `{"path":"README.md"}` {
		t.Fatalf("tool use = %#v", use)
	}
	result := got.Messages[2].Blocks[0]
	if result.Type != "tool_result" || result.ToolUseID != "call_1" || result.Content != "hello" {
		t.Fatalf("tool result = %#v", result)
	}
}

func TestAnthropicToolStreamBuildsToolCall(t *testing.T) {
	events := make(chan Event, 5)
	stream := &MessageStream{Events: events}
	index := 0
	partial1, _ := json.Marshal(Delta{Type: "input_json_delta", PartialJSON: `{"path":"`})
	partial2, _ := json.Marshal(Delta{Type: "input_json_delta", PartialJSON: `README.md"}`})
	stop, _ := json.Marshal(Delta{StopReason: "tool_use"})
	events <- Event{Type: "message_start", Message: &MessageStart{ID: "msg_1", Model: "claude-test"}}
	events <- Event{Type: "content_block_start", Index: &index, ContentBlock: &ContentBlock{Type: "tool_use", ID: "call_1", Name: "read_file"}}
	events <- Event{Type: "content_block_delta", Index: &index, Delta: partial1}
	events <- Event{Type: "content_block_delta", Index: &index, Delta: partial2}
	events <- Event{Type: "message_delta", Delta: stop}
	close(events)

	builder := openai.NewMessageBuilder()
	finish := ""
	for chunk := range AnthropicStreamToChatCompletionStream(stream, nil) {
		if message := chunk.GetMessage(); message != nil {
			builder.Update(*message)
		}
		if choice := chunk.GetFirstChoice(); choice != nil && choice.FinishReason != "" {
			finish = choice.FinishReason
		}
	}
	message := builder.Build()
	if len(message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %#v", message.ToolCalls)
	}
	call := message.ToolCalls[0]
	if call.ID != "call_1" || call.Function.Name != "read_file" || call.Function.Arguments != `{"path":"README.md"}` {
		t.Fatalf("tool call = %#v", call)
	}
	if finish != "tool_calls" {
		t.Fatalf("finish reason = %q", finish)
	}
}
