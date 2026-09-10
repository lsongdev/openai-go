package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/lsongdev/miya-agents/openai"
)

// NewAnthropicRequestFromChatCompletionRequest converts the agent's OpenAI-shaped
// conversation into a native Anthropic Messages request.
func NewAnthropicRequestFromChatCompletionRequest(req *openai.ChatCompletionRequest) *Request {
	out := &Request{
		Model:         req.Model,
		Stream:        req.Stream,
		TopP:          req.TopP,
		Temperature:   req.Temperature,
		StopSequences: req.Stop,
		MaxTokens:     req.MaxTokens,
	}
	if out.MaxTokens == 0 {
		out.MaxTokens = 4096
	}
	for _, tool := range req.Tools {
		out.Tools = append(out.Tools, Tool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: tool.Function.Parameters,
		})
	}

	var system []string
	for _, msg := range req.Messages {
		switch msg.Role {
		case openai.RoleSystem:
			system = append(system, msg.Content)
		case openai.RoleUser:
			out.Messages = append(out.Messages, TextMessage("user", msg.Content))
		case openai.RoleAssistant:
			blocks := make([]ContentBlock, 0, 1+len(msg.ToolCalls))
			if msg.Content != "" {
				blocks = append(blocks, ContentBlock{Type: "text", Text: msg.Content})
			}
			for _, call := range msg.ToolCalls {
				input := json.RawMessage(call.Function.Arguments)
				if len(input) == 0 {
					input = json.RawMessage(`{}`)
				}
				blocks = append(blocks, ContentBlock{
					Type:  "tool_use",
					ID:    call.ID,
					Name:  call.Function.Name,
					Input: input,
				})
			}
			if len(blocks) > 0 {
				out.Messages = append(out.Messages, Message{Role: "assistant", Blocks: blocks})
			}
		case openai.RoleTool:
			block := ContentBlock{
				Type:      "tool_result",
				ToolUseID: msg.ToolCallID,
				Content:   msg.Content,
			}
			last := len(out.Messages) - 1
			if last >= 0 && toolResultsOnly(out.Messages[last]) {
				out.Messages[last].Blocks = append(out.Messages[last].Blocks, block)
			} else {
				out.Messages = append(out.Messages, Message{Role: "user", Blocks: []ContentBlock{block}})
			}
		}
	}
	out.System = strings.Join(system, "\n\n")
	return out
}

func toolResultsOnly(message Message) bool {
	if message.Role != "user" || len(message.Blocks) == 0 {
		return false
	}
	for _, block := range message.Blocks {
		if block.Type != "tool_result" {
			return false
		}
	}
	return true
}

// NewAnthropicResponseFromChatCompletionResponse converts an OpenAI chat response
// to Anthropic's response shape.
func NewAnthropicResponseFromChatCompletionResponse(resp *openai.ChatCompletionResponse) *Response {
	out := &Response{ID: resp.ID, Type: "message", Role: "assistant", Model: resp.Model}
	if choice := resp.GetFirstChoice(); choice != nil {
		if msg := choice.Message; msg != nil {
			if msg.Content != "" {
				out.Content = append(out.Content, ContentBlock{Type: "text", Text: msg.Content})
			}
			if msg.ReasoningContent != "" {
				out.Content = append(out.Content, ContentBlock{Type: "thinking", Thinking: msg.ReasoningContent})
			}
			for _, call := range msg.ToolCalls {
				input := json.RawMessage(call.Function.Arguments)
				if len(input) == 0 {
					input = json.RawMessage(`{}`)
				}
				out.Content = append(out.Content, ContentBlock{
					Type: "tool_use", ID: call.ID, Name: call.Function.Name, Input: input,
				})
			}
		}
		out.StopReason = MapStopReasonReverse(choice.FinishReason)
	}
	if resp.Usage != nil {
		out.Usage = Usage{InputTokens: resp.Usage.PromptTokens, OutputTokens: resp.Usage.CompletionTokens}
	}
	return out
}

// NewChatCompletionResponseFromAnthropicResponse converts an Anthropic response
// into the shape consumed by the agent loop.
func NewChatCompletionResponseFromAnthropicResponse(resp *Response) *openai.ChatCompletionResponse {
	message := openai.ChatCompletionMessage{Role: openai.RoleAssistant}
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			message.Content += block.Text
		case "thinking":
			message.ReasoningContent += block.Thinking
		case "tool_use":
			args := string(block.Input)
			if args == "" {
				args = `{}`
			}
			message.ToolCalls = append(message.ToolCalls, openai.ToolCall{
				Index: len(message.ToolCalls),
				ID:    block.ID,
				Type:  "function",
				Function: openai.FunctionCall{
					Name: block.Name, Arguments: args,
				},
			})
		}
	}
	out := openai.NewChatCompletionResponse(resp.ID, resp.Model, message.Content, message.ReasoningContent)
	out.Choices[0].Message = &message
	out.Choices[0].FinishReason = MapStopReason(resp.StopReason)
	out.Usage = &openai.CompletionUsage{
		PromptTokens:     resp.Usage.InputTokens,
		CompletionTokens: resp.Usage.OutputTokens,
		TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
	}
	return out
}

// NewChatCompletionRequestFromAnthropicRequest converts an Anthropic request to
// OpenAI chat-completion format.
func NewChatCompletionRequestFromAnthropicRequest(req *Request) *openai.ChatCompletionRequest {
	out := &openai.ChatCompletionRequest{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Stream:      req.Stream,
		TopP:        req.TopP,
		Stop:        req.StopSequences,
		Temperature: req.Temperature,
	}
	if req.System != "" {
		out.Messages = append(out.Messages, openai.SystemMessage(req.System))
	}
	for _, tool := range req.Tools {
		out.Tools = append(out.Tools, openai.ToolDef{
			Type: "function",
			Function: openai.FunctionDef{
				Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema,
			},
		})
	}
	for _, msg := range req.Messages {
		switch msg.Role {
		case "assistant":
			converted := openai.ChatCompletionMessage{Role: openai.RoleAssistant}
			for _, block := range msg.contentBlocks() {
				switch block.Type {
				case "text":
					converted.Content += block.Text
				case "thinking":
					converted.ReasoningContent += block.Thinking
				case "tool_use":
					args := string(block.Input)
					if args == "" {
						args = `{}`
					}
					converted.ToolCalls = append(converted.ToolCalls, openai.ToolCall{
						Index: len(converted.ToolCalls), ID: block.ID, Type: "function",
						Function: openai.FunctionCall{Name: block.Name, Arguments: args},
					})
				}
			}
			out.Messages = append(out.Messages, converted)
		case "user":
			var text strings.Builder
			for _, block := range msg.contentBlocks() {
				switch block.Type {
				case "text":
					text.WriteString(block.Text)
				case "tool_result":
					out.Messages = append(out.Messages, openai.ToolResultMessage(block.ToolUseID, "", block.Content))
				}
			}
			if text.Len() > 0 {
				out.Messages = append(out.Messages, openai.UserMessage(text.String()))
			}
		}
	}
	return out
}

// MapStopReason maps Anthropic stop reasons to OpenAI finish reasons.
func MapStopReason(reason string) string {
	switch reason {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	default:
		return reason
	}
}

func (c *Client) CreateChatCompletion(ctx context.Context, req *openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	resp, err := c.CreateMessage(ctx, NewAnthropicRequestFromChatCompletionRequest(req))
	if err != nil {
		return nil, err
	}
	return NewChatCompletionResponseFromAnthropicResponse(resp), nil
}

func (c *Client) CreateChatCompletionStream(ctx context.Context, req *openai.ChatCompletionRequest) (<-chan openai.ChatCompletionResponse, error) {
	anthReq := NewAnthropicRequestFromChatCompletionRequest(req)
	anthReq.Stream = true
	stream, err := c.CreateMessageStream(ctx, anthReq)
	if err != nil {
		return nil, err
	}
	return AnthropicStreamToChatCompletionStream(stream, nil), nil
}

// AnthropicStreamToChatCompletionStream converts Anthropic SSE events to the
// stream shape consumed by the agent loop.
func AnthropicStreamToChatCompletionStream(stream *MessageStream, onEvent func(Event)) <-chan openai.ChatCompletionResponse {
	ch := make(chan openai.ChatCompletionResponse)
	go func() {
		defer close(ch)
		var messageID, model string
		var inputTokens int
		for event := range stream.Events {
			if onEvent != nil {
				onEvent(event)
			}
			switch event.Type {
			case "error":
				if event.Error != nil {
					ch <- openai.ChatCompletionResponse{Error: &openai.Error{Type: event.Error.Type, Message: event.Error.Message}}
				}
				return
			case "message_start":
				if event.Message != nil {
					messageID = event.Message.ID
					model = event.Message.Model
					inputTokens = event.Message.Usage.InputTokens
				}
			case "content_block_start":
				if event.ContentBlock == nil || event.ContentBlock.Type != "tool_use" {
					continue
				}
				index := 0
				if event.Index != nil {
					index = *event.Index
				}
				ch <- chatDelta(messageID, model, openai.ChatCompletionMessage{
					ToolCalls: []openai.ToolCall{{
						Index: index, ID: event.ContentBlock.ID, Type: "function",
						Function: openai.FunctionCall{Name: event.ContentBlock.Name},
					}},
				})
			case "content_block_delta":
				var delta Delta
				if err := json.Unmarshal(event.Delta, &delta); err != nil {
					continue
				}
				message := openai.ChatCompletionMessage{}
				switch delta.Type {
				case "text_delta":
					message.Content = delta.Text
				case "thinking_delta":
					message.ReasoningContent = delta.Thinking
				case "input_json_delta":
					index := 0
					if event.Index != nil {
						index = *event.Index
					}
					message.ToolCalls = []openai.ToolCall{{
						Index: index,
						Function: openai.FunctionCall{Arguments: delta.PartialJSON},
					}}
				}
				if !message.IsEmpty() {
					ch <- chatDelta(messageID, model, message)
				}
			case "message_delta":
				var delta Delta
				if err := json.Unmarshal(event.Delta, &delta); err != nil || delta.StopReason == "" {
					continue
				}
				outputTokens := 0
				if event.Usage != nil {
					outputTokens = event.Usage.OutputTokens
				}
				ch <- openai.ChatCompletionResponse{
					ID: messageID, Model: model, Object: "chat.completion.chunk",
					Choices: []openai.ChatCompletionChoice{{Index: 0, FinishReason: MapStopReason(delta.StopReason)}},
					Usage: &openai.CompletionUsage{
						PromptTokens: inputTokens, CompletionTokens: outputTokens, TotalTokens: inputTokens + outputTokens,
					},
				}
			case "message_stop":
				return
			}
		}
	}()
	return ch
}

func chatDelta(id, model string, message openai.ChatCompletionMessage) openai.ChatCompletionResponse {
	return openai.ChatCompletionResponse{
		ID: id, Model: model, Object: "chat.completion.chunk",
		Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &message}},
	}
}

// OpenAIStreamToAnthropicStream converts an OpenAI chat completion stream to Anthropic SSE events.
// The onChunk callback receives each chunk before it is processed (useful for assembling the final response).
// Returns the final assembled ChatCompletionResponse after the stream completes.
func OpenAIStreamToAnthropicStream(chunks <-chan openai.ChatCompletionResponse, w http.ResponseWriter, onChunk func(openai.ChatCompletionResponse)) *openai.ChatCompletionResponse {
	anthStream := NewResponseWriter(w)

	var (
		messageID        string
		model            string
		stopReason       string
		hasContent       bool
		sentMessageStart bool
		notedFinish      bool
	)

	assembler := openai.NewResponseAssembler()

	sendMessageStart := func() {
		if sentMessageStart {
			return
		}
		anthStream.SendMessageStart(messageID, "message", "assistant", model)
		sentMessageStart = true
	}

	sendContentBlockStart := func() {
		anthStream.SendContentBlockStart(0, "text")
		hasContent = true
	}

	sendFinish := func() {
		if notedFinish {
			return
		}
		notedFinish = true
		if hasContent {
			anthStream.SendContentBlockStop(0)
		}
		resp := assembler.Build()
		var usageOut *Usage
		if resp.Usage != nil {
			usageOut = &Usage{OutputTokens: resp.Usage.CompletionTokens}
		}
		anthStream.SendMessageDelta(stopReason, nil, usageOut)
		anthStream.SendMessageStop()
	}

	for chunk := range chunks {
		assembler.Update(chunk)
		if onChunk != nil {
			onChunk(chunk)
		}

		if chunk.ID != "" && messageID == "" {
			messageID = chunk.ID
		}
		if chunk.Model != "" && model == "" {
			model = chunk.Model
		}

		if len(chunk.Choices) == 0 {
			if chunk.Usage != nil && chunk.Usage.TotalTokens > 0 && !notedFinish {
				if stopReason == "" {
					stopReason = "end_turn"
				}
				sendMessageStart()
				sendFinish()
			}
			continue
		}

		choice := chunk.Choices[0]
		delta := choice.Delta

		if delta != nil && delta.Content != "" {
			sendMessageStart()
			if !hasContent {
				sendContentBlockStart()
			}
			anthStream.SendContentBlockDelta(0, Delta{Type: "text_delta", Text: delta.Content})
		}

		if choice.FinishReason != "" {
			stopReason = MapStopReasonReverse(choice.FinishReason)
			sendMessageStart()
			sendFinish()
		}
	}

	if !notedFinish {
		if stopReason == "" {
			stopReason = "end_turn"
		}
		if !sentMessageStart {
			sendMessageStart()
		}
		sendFinish()
	}

	return assembler.Build()
}

func (c *Client) CreateEmbeddings(ctx context.Context, req *openai.EmbeddingRequest) (*openai.EmbeddingResponse, error) {
	return nil, fmt.Errorf("embeddings not supported by anthropic client")
}

// MapStopReasonReverse maps OpenAI finish reasons to Anthropic stop reasons.
func MapStopReasonReverse(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return reason
	}
}
