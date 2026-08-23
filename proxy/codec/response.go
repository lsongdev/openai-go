package codec

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"
)

var responseIDCounter atomic.Uint64

func DecodeResponse(protocol Protocol, body []byte) (*Response, []Diagnostic, error) {
	value, err := decodeObject(body)
	if err != nil {
		return nil, nil, err
	}
	switch protocol {
	case OpenAIChat:
		return decodeChatResponse(value)
	case OpenAIResponses:
		return decodeResponsesResponse(value)
	case Anthropic:
		return decodeAnthropicResponse(value)
	default:
		return nil, nil, fmt.Errorf("unsupported response protocol %q", protocol)
	}
}

func EncodeResponse(protocol Protocol, response *Response) ([]byte, []Diagnostic, error) {
	var value map[string]any
	var diagnostics []Diagnostic
	switch protocol {
	case OpenAIChat:
		value, diagnostics = encodeChatResponse(response)
	case OpenAIResponses:
		value, diagnostics = encodeResponsesResponse(response)
	case Anthropic:
		value, diagnostics = encodeAnthropicResponse(response)
	default:
		return nil, nil, fmt.Errorf("unsupported response protocol %q", protocol)
	}
	body, err := marshalObject(value)
	return body, diagnostics, err
}

func decodeAnthropicResponse(value map[string]any) (*Response, []Diagnostic, error) {
	response := &Response{}
	diagnostics := diagnoseRequestKeys(value,
		[]string{"id", "type", "role", "model", "content", "usage", "stop_reason", "stop_sequence"},
		map[string]string{"container": "container state is provider-specific", "context_management": "context management metadata is provider-specific", "stop_details": "detailed stop metadata is provider-specific"})
	response.ID, _ = stringValue(value["id"])
	response.Model, _ = stringValue(value["model"])
	stopReason, _ := stringValue(value["stop_reason"])
	response.StopReason = anthropicStopReason(stopReason)
	items, ok := array(value["content"])
	if !ok {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.content", Message: "must be an array"})
	} else {
		for index, raw := range items {
			item, ok := object(raw)
			path := fmt.Sprintf("$.content[%d]", index)
			if !ok {
				diagnostics = append(diagnostics, Diagnostic{Path: path, Message: "must be an object"})
				continue
			}
			typeName, _ := stringValue(item["type"])
			switch typeName {
			case "text":
				text, ok := stringValue(item["text"])
				if ok {
					response.Content = append(response.Content, Content{Type: ContentText, Text: text})
				}
			case "tool_use":
				id, _ := stringValue(item["id"])
				name, _ := stringValue(item["name"])
				arguments, argsOK := parseArguments(item["input"])
				if id == "" || name == "" || !argsOK {
					diagnostics = append(diagnostics, Diagnostic{Path: path, Message: "invalid tool_use block"})
					continue
				}
				response.Content = append(response.Content, Content{Type: ContentToolCall, ID: id, Name: name, Arguments: arguments})
			case "thinking":
				thinking, _ := stringValue(item["thinking"])
				response.Content = append(response.Content, Content{Type: ContentReasoning, Text: thinking})
			case "redacted_thinking":
				// Redacted reasoning has no portable payload.
			default:
				diagnostics = append(diagnostics, Diagnostic{Path: path + ".type", Message: fmt.Sprintf("content type %q is not portable", typeName)})
			}
		}
	}
	response.Usage = decodeAnthropicUsage(value["usage"])
	return response, diagnostics, nil
}

func decodeResponsesResponse(value map[string]any) (*Response, []Diagnostic, error) {
	response := &Response{}
	diagnostics := diagnoseRequestKeys(value,
		[]string{"id", "object", "created_at", "status", "error", "incomplete_details", "instructions", "model", "output", "usage"},
		map[string]string{
			"parallel_tool_calls": "parallel tool metadata is provider-specific", "previous_response_id": "provider-side conversation state is not portable",
			"temperature": "sampling metadata is not portable", "top_p": "sampling metadata is not portable", "max_output_tokens": "output limit metadata is not portable",
			"tool_choice": "tool execution metadata is not portable", "tools": "tool definitions are request metadata", "metadata": "metadata is not portable",
			"user": "user attribution is not portable", "text": "text configuration is provider-specific", "reasoning": "reasoning configuration is provider-specific",
			"background": "background execution state is provider-specific", "service_tier": "service tier is provider-specific", "prompt_cache_key": "prompt cache routing is provider-specific",
		})
	response.ID, _ = stringValue(value["id"])
	response.Model, _ = stringValue(value["model"])
	response.CreatedAt, _ = int64Value(value["created_at"])
	status, _ := stringValue(value["status"])
	response.StopReason = responsesStopReason(status, value["incomplete_details"])
	items, ok := array(value["output"])
	if !ok {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.output", Message: "must be an array"})
	} else {
		for index, raw := range items {
			item, ok := object(raw)
			path := fmt.Sprintf("$.output[%d]", index)
			if !ok {
				diagnostics = append(diagnostics, Diagnostic{Path: path, Message: "must be an object"})
				continue
			}
			typeName, _ := stringValue(item["type"])
			switch typeName {
			case "message":
				response.Content = append(response.Content, decodeResponsesOutputContent(item["content"], path+".content", &diagnostics)...)
			case "function_call":
				id, _ := stringValue(item["call_id"])
				if id == "" {
					id, _ = stringValue(item["id"])
				}
				name, _ := stringValue(item["name"])
				arguments, argsOK := parseArguments(item["arguments"])
				if id == "" || name == "" || !argsOK {
					diagnostics = append(diagnostics, Diagnostic{Path: path, Message: "invalid function_call output"})
					continue
				}
				response.Content = append(response.Content, Content{Type: ContentToolCall, ID: id, Name: name, Arguments: arguments})
			case "reasoning":
				if summaries, ok := array(item["summary"]); ok {
					for _, rawSummary := range summaries {
						summary, _ := object(rawSummary)
						if text, ok := stringValue(summary["text"]); ok {
							response.Content = append(response.Content, Content{Type: ContentReasoning, Text: text})
						}
					}
				}
			default:
				diagnostics = append(diagnostics, Diagnostic{Path: path + ".type", Message: fmt.Sprintf("output type %q is not portable", typeName)})
			}
		}
	}
	response.Usage = decodeResponsesUsage(value["usage"])
	if response.StopReason == "stop" {
		for _, content := range response.Content {
			if content.Type == ContentToolCall {
				response.StopReason = "tool_use"
				break
			}
		}
	}
	return response, diagnostics, nil
}

func decodeResponsesOutputContent(value any, path string, diagnostics *[]Diagnostic) []Content {
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: path, Message: "must be an array"})
		return nil
	}
	var contents []Content
	for index, raw := range items {
		item, ok := object(raw)
		itemPath := fmt.Sprintf("%s[%d]", path, index)
		if !ok {
			continue
		}
		typeName, _ := stringValue(item["type"])
		switch typeName {
		case "output_text":
			text, _ := stringValue(item["text"])
			contents = append(contents, Content{Type: ContentText, Text: text})
		case "refusal":
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath + ".type", Message: "refusal content is not portable as ordinary text"})
		default:
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath + ".type", Message: fmt.Sprintf("content type %q is not portable", typeName)})
		}
	}
	return contents
}

func decodeChatResponse(value map[string]any) (*Response, []Diagnostic, error) {
	response := &Response{}
	diagnostics := diagnoseRequestKeys(value,
		[]string{"id", "object", "created", "model", "choices", "usage"},
		map[string]string{"system_fingerprint": "system fingerprint is provider telemetry", "service_tier": "service tier is provider-specific"})
	response.ID, _ = stringValue(value["id"])
	response.Model, _ = stringValue(value["model"])
	response.CreatedAt, _ = int64Value(value["created"])
	choices, ok := array(value["choices"])
	if !ok || len(choices) != 1 {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.choices", Message: "exactly one choice is required for translation"})
		return response, diagnostics, nil
	}
	choice, _ := object(choices[0])
	message, ok := object(choice["message"])
	if !ok {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.choices[0].message", Message: "must be an object"})
		return response, diagnostics, nil
	}
	response.Content = append(response.Content, decodeChatResponseText(message["content"], &diagnostics)...)
	if calls, ok := array(message["tool_calls"]); ok {
		for index, raw := range calls {
			call, _ := object(raw)
			fn, _ := object(call["function"])
			id, _ := stringValue(call["id"])
			name, _ := stringValue(fn["name"])
			arguments, argsOK := parseArguments(fn["arguments"])
			if id == "" || name == "" || !argsOK {
				diagnostics = append(diagnostics, Diagnostic{Path: fmt.Sprintf("$.choices[0].message.tool_calls[%d]", index), Message: "invalid function tool call"})
				continue
			}
			response.Content = append(response.Content, Content{Type: ContentToolCall, ID: id, Name: name, Arguments: arguments})
		}
	}
	finishReason, _ := stringValue(choice["finish_reason"])
	response.StopReason = chatStopReason(finishReason)
	response.Usage = decodeChatUsage(value["usage"])
	return response, diagnostics, nil
}

func decodeChatResponseText(value any, diagnostics *[]Diagnostic) []Content {
	if value == nil {
		return nil
	}
	if text, ok := stringValue(value); ok {
		if text == "" {
			return nil
		}
		return []Content{{Type: ContentText, Text: text}}
	}
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: "$.choices[0].message.content", Message: "must be text"})
		return nil
	}
	var text strings.Builder
	for _, raw := range items {
		item, ok := object(raw)
		if !ok || item["type"] != "text" {
			*diagnostics = append(*diagnostics, Diagnostic{Path: "$.choices[0].message.content", Message: "only text content is portable"})
			continue
		}
		part, _ := stringValue(item["text"])
		text.WriteString(part)
	}
	if text.Len() == 0 {
		return nil
	}
	return []Content{{Type: ContentText, Text: text.String()}}
}

func encodeAnthropicResponse(response *Response) (map[string]any, []Diagnostic) {
	var content []any
	var diagnostics []Diagnostic
	for _, item := range response.Content {
		switch item.Type {
		case ContentText:
			content = append(content, map[string]any{"type": "text", "text": item.Text})
		case ContentToolCall:
			content = append(content, map[string]any{"type": "tool_use", "id": item.ID, "name": item.Name, "input": item.Arguments})
		case ContentReasoning:
			content = append(content, map[string]any{"type": "thinking", "thinking": item.Text, "signature": ""})
		case ContentToolResult:
			diagnostics = append(diagnostics, Diagnostic{Path: "$.content", Message: "assistant responses cannot contain tool results"})
		}
	}
	value := map[string]any{
		"id":            ensureID(response.ID, "msg"),
		"type":          "message",
		"role":          "assistant",
		"model":         response.Model,
		"content":       content,
		"stop_reason":   encodeAnthropicStopReason(response.StopReason),
		"stop_sequence": nil,
	}
	if response.Usage != nil {
		value["usage"] = map[string]any{
			"input_tokens":                response.Usage.InputTokens,
			"output_tokens":               response.Usage.OutputTokens,
			"cache_read_input_tokens":     response.Usage.CachedInputTokens,
			"cache_creation_input_tokens": response.Usage.CacheWriteInputTokens,
		}
	}
	return value, diagnostics
}

func encodeResponsesResponse(response *Response) (map[string]any, []Diagnostic) {
	responseID := ensureID(response.ID, "resp")
	var output []any
	for index, item := range response.Content {
		switch item.Type {
		case ContentText:
			output = append(output, map[string]any{
				"type": "message", "id": fmt.Sprintf("msg_%d", index), "status": "completed", "role": "assistant",
				"content": []any{map[string]any{"type": "output_text", "text": item.Text, "annotations": []any{}}},
			})
		case ContentToolCall:
			arguments, _ := json.Marshal(item.Arguments)
			output = append(output, map[string]any{
				"type": "function_call", "id": fmt.Sprintf("fc_%d", index), "status": "completed",
				"call_id": item.ID, "name": item.Name, "arguments": string(arguments),
			})
		case ContentReasoning:
			output = append(output, map[string]any{"type": "reasoning", "id": fmt.Sprintf("rs_%d", index), "summary": []any{map[string]any{"type": "summary_text", "text": item.Text}}})
		}
	}
	status := "completed"
	var incomplete any
	if response.StopReason == "length" {
		status = "incomplete"
		incomplete = map[string]any{"reason": "max_output_tokens"}
	} else if response.StopReason == "error" {
		status = "failed"
	}
	value := map[string]any{
		"id": responseID, "object": "response", "created_at": response.CreatedAt,
		"status": status, "error": nil, "incomplete_details": incomplete,
		"instructions": nil, "model": response.Model, "output": output,
		"parallel_tool_calls": true, "tools": []any{}, "tool_choice": "auto",
	}
	if value["created_at"].(int64) == 0 {
		value["created_at"] = time.Now().Unix()
	}
	if response.Usage != nil {
		value["usage"] = map[string]any{
			"input_tokens": response.Usage.InputTokens, "output_tokens": response.Usage.OutputTokens, "total_tokens": response.Usage.TotalTokens,
			"input_tokens_details":  map[string]any{"cached_tokens": response.Usage.CachedInputTokens},
			"output_tokens_details": map[string]any{"reasoning_tokens": response.Usage.ReasoningTokens},
		}
	}
	return value, nil
}

func encodeChatResponse(response *Response) (map[string]any, []Diagnostic) {
	var text strings.Builder
	var reasoning strings.Builder
	var calls []any
	var diagnostics []Diagnostic
	for _, item := range response.Content {
		switch item.Type {
		case ContentText:
			text.WriteString(item.Text)
		case ContentToolCall:
			arguments, _ := json.Marshal(item.Arguments)
			calls = append(calls, map[string]any{"id": item.ID, "type": "function", "function": map[string]any{"name": item.Name, "arguments": string(arguments)}})
		case ContentReasoning:
			reasoning.WriteString(item.Text)
		case ContentToolResult:
			diagnostics = append(diagnostics, Diagnostic{Path: "$.content", Message: "assistant responses cannot contain tool results"})
		}
	}
	message := map[string]any{"role": "assistant", "content": text.String()}
	if reasoning.Len() > 0 {
		message["reasoning_content"] = reasoning.String()
	}
	if len(calls) > 0 {
		message["tool_calls"] = calls
		if text.Len() == 0 {
			message["content"] = nil
		}
	}
	value := map[string]any{
		"id": ensureID(response.ID, "chatcmpl"), "object": "chat.completion", "created": response.CreatedAt,
		"model": response.Model, "choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": encodeChatStopReason(response.StopReason)}},
	}
	if value["created"].(int64) == 0 {
		value["created"] = time.Now().Unix()
	}
	if response.Usage != nil {
		value["usage"] = map[string]any{
			"prompt_tokens": response.Usage.InputTokens, "completion_tokens": response.Usage.OutputTokens, "total_tokens": response.Usage.TotalTokens,
			"prompt_tokens_details":     map[string]any{"cached_tokens": response.Usage.CachedInputTokens},
			"completion_tokens_details": map[string]any{"reasoning_tokens": response.Usage.ReasoningTokens},
		}
	}
	return value, diagnostics
}

func decodeAnthropicUsage(value any) *Usage {
	usage, ok := object(value)
	if !ok {
		return nil
	}
	input, _ := intValue(usage["input_tokens"])
	output, _ := intValue(usage["output_tokens"])
	cached, _ := intValue(usage["cache_read_input_tokens"])
	cacheWrite, _ := intValue(usage["cache_creation_input_tokens"])
	return &Usage{InputTokens: input, OutputTokens: output, TotalTokens: input + output, CachedInputTokens: cached, CacheWriteInputTokens: cacheWrite}
}

func decodeResponsesUsage(value any) *Usage {
	usage, ok := object(value)
	if !ok {
		return nil
	}
	input, _ := intValue(usage["input_tokens"])
	output, _ := intValue(usage["output_tokens"])
	total, _ := intValue(usage["total_tokens"])
	inputDetails, _ := object(usage["input_tokens_details"])
	outputDetails, _ := object(usage["output_tokens_details"])
	cached, _ := intValue(inputDetails["cached_tokens"])
	reasoning, _ := intValue(outputDetails["reasoning_tokens"])
	return &Usage{InputTokens: input, OutputTokens: output, TotalTokens: total, CachedInputTokens: cached, ReasoningTokens: reasoning}
}

func decodeChatUsage(value any) *Usage {
	usage, ok := object(value)
	if !ok {
		return nil
	}
	input, _ := intValue(usage["prompt_tokens"])
	output, _ := intValue(usage["completion_tokens"])
	total, _ := intValue(usage["total_tokens"])
	inputDetails, _ := object(usage["prompt_tokens_details"])
	outputDetails, _ := object(usage["completion_tokens_details"])
	cached, _ := intValue(inputDetails["cached_tokens"])
	reasoning, _ := intValue(outputDetails["reasoning_tokens"])
	return &Usage{InputTokens: input, OutputTokens: output, TotalTokens: total, CachedInputTokens: cached, ReasoningTokens: reasoning}
}

func int64Value(value any) (int64, bool) {
	switch number := value.(type) {
	case json.Number:
		parsed, err := number.Int64()
		return parsed, err == nil
	case float64:
		return int64(number), number == float64(int64(number))
	case int64:
		return number, true
	default:
		return 0, false
	}
}

func anthropicStopReason(value string) string {
	switch value {
	case "max_tokens", "model_context_window_exceeded":
		return "length"
	case "tool_use":
		return "tool_use"
	default:
		return "stop"
	}
}

func responsesStopReason(status string, incomplete any) string {
	switch status {
	case "failed":
		return "error"
	case "cancelled":
		return "cancelled"
	case "incomplete":
		if details, ok := object(incomplete); ok {
			if reason, _ := stringValue(details["reason"]); reason == "max_output_tokens" {
				return "length"
			}
		}
		return "length"
	default:
		return "stop"
	}
}

func chatStopReason(value string) string {
	switch value {
	case "length":
		return "length"
	case "tool_calls", "function_call":
		return "tool_use"
	case "content_filter":
		return "content_filter"
	default:
		return "stop"
	}
}

func encodeAnthropicStopReason(value string) string {
	switch value {
	case "length":
		return "max_tokens"
	case "tool_use":
		return "tool_use"
	default:
		return "end_turn"
	}
}

func encodeChatStopReason(value string) string {
	switch value {
	case "length":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "content_filter":
		return "content_filter"
	default:
		return "stop"
	}
}

func ensureID(value, prefix string) string {
	if value != "" && strings.HasPrefix(value, prefix+"_") {
		return value
	}
	return fmt.Sprintf("%s_%d_%d", prefix, time.Now().UnixNano(), responseIDCounter.Add(1))
}
