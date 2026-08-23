package codec

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

func DecodeRequest(protocol Protocol, body []byte) (*Request, []Diagnostic, error) {
	value, err := decodeObject(body)
	if err != nil {
		return nil, nil, err
	}
	switch protocol {
	case OpenAIChat:
		return decodeChatRequest(value)
	case OpenAIResponses:
		return decodeResponsesRequest(value)
	case Anthropic:
		return decodeAnthropicRequest(value)
	default:
		return nil, nil, fmt.Errorf("unsupported request protocol %q", protocol)
	}
}

func EncodeRequest(protocol Protocol, request *Request) ([]byte, []Diagnostic, error) {
	var value map[string]any
	var diagnostics []Diagnostic
	switch protocol {
	case OpenAIChat:
		value, diagnostics = encodeChatRequest(request)
	case OpenAIResponses:
		value, diagnostics = encodeResponsesRequest(request)
	case Anthropic:
		value, diagnostics = encodeAnthropicRequest(request)
	default:
		return nil, nil, fmt.Errorf("unsupported request protocol %q", protocol)
	}
	body, err := marshalObject(value)
	return body, diagnostics, err
}

func baseRequest(value map[string]any) (*Request, []Diagnostic) {
	request := &Request{}
	var diagnostics []Diagnostic
	if model, ok := stringValue(value["model"]); ok && model != "" {
		request.Model = model
	} else {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.model", Message: "must be a non-empty string"})
	}
	if stream, ok := boolValue(value["stream"]); ok {
		request.Stream = stream
	} else if value["stream"] != nil {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.stream", Message: "must be a boolean"})
	}
	if temperature, ok := floatValue(value["temperature"]); ok {
		request.Temperature = temperature
	} else if value["temperature"] != nil {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.temperature", Message: "must be a number"})
	}
	if topP, ok := floatValue(value["top_p"]); ok {
		request.TopP = topP
	} else if value["top_p"] != nil {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.top_p", Message: "must be a number"})
	}
	return request, diagnostics
}

func decodeAnthropicRequest(value map[string]any) (*Request, []Diagnostic, error) {
	request, diagnostics := baseRequest(value)
	diagnostics = append(diagnostics, diagnoseRequestKeys(value,
		[]string{"model", "max_tokens", "messages", "system", "tools", "tool_choice", "temperature", "top_p", "stop_sequences", "stream"},
		map[string]string{
			"metadata": "client attribution is not portable", "service_tier": "service tier is provider-specific",
			"thinking": "thinking configuration is provider-specific", "context_management": "context management is provider-specific",
			"output_config": "output configuration is provider-specific",
		})...)
	if maxTokens, ok := intValue(value["max_tokens"]); ok && maxTokens > 0 {
		request.MaxOutputTokens = maxTokens
	} else {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.max_tokens", Message: "must be a positive integer"})
	}
	if stops, ok := stringsValue(value["stop_sequences"]); ok {
		request.Stop = stops
	} else if value["stop_sequences"] != nil {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.stop_sequences", Message: "must be an array of strings"})
	}
	request.Messages = append(request.Messages, decodeAnthropicSystem(value["system"], &diagnostics)...)
	items, ok := array(value["messages"])
	if !ok {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.messages", Message: "must be an array"})
	} else {
		for index, raw := range items {
			item, ok := object(raw)
			if !ok {
				diagnostics = append(diagnostics, Diagnostic{Path: fmt.Sprintf("$.messages[%d]", index), Message: "must be an object"})
				continue
			}
			role, _ := stringValue(item["role"])
			if role != "user" && role != "assistant" {
				diagnostics = append(diagnostics, Diagnostic{Path: fmt.Sprintf("$.messages[%d].role", index), Message: "must be user or assistant"})
				continue
			}
			contents := decodeAnthropicContent(item["content"], fmt.Sprintf("$.messages[%d].content", index), &diagnostics)
			request.Messages = append(request.Messages, Message{Role: role, Content: contents})
		}
	}
	request.Tools = decodeAnthropicTools(value["tools"], &diagnostics)
	request.ToolChoice = decodeAnthropicToolChoice(value["tool_choice"], &diagnostics)
	return request, diagnostics, nil
}

func decodeAnthropicSystem(value any, diagnostics *[]Diagnostic) []Message {
	if value == nil {
		return nil
	}
	if text, ok := stringValue(value); ok {
		return []Message{{Role: "system", Content: []Content{{Type: ContentText, Text: text}}}}
	}
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: "$.system", Message: "must be a string or text block array"})
		return nil
	}
	var text strings.Builder
	for index, raw := range items {
		item, ok := object(raw)
		if !ok || item["type"] != "text" {
			*diagnostics = append(*diagnostics, Diagnostic{Path: fmt.Sprintf("$.system[%d]", index), Message: "only text blocks are portable"})
			continue
		}
		if item["cache_control"] != nil {
			*diagnostics = append(*diagnostics, Warning(fmt.Sprintf("$.system[%d].cache_control", index), "prompt cache control is provider-specific"))
		}
		part, ok := stringValue(item["text"])
		if !ok {
			*diagnostics = append(*diagnostics, Diagnostic{Path: fmt.Sprintf("$.system[%d].text", index), Message: "must be a string"})
			continue
		}
		text.WriteString(part)
	}
	if text.Len() == 0 {
		return nil
	}
	return []Message{{Role: "system", Content: []Content{{Type: ContentText, Text: text.String()}}}}
}

func decodeAnthropicContent(value any, path string, diagnostics *[]Diagnostic) []Content {
	if text, ok := stringValue(value); ok {
		return []Content{{Type: ContentText, Text: text}}
	}
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: path, Message: "must be a string or content block array"})
		return nil
	}
	contents := make([]Content, 0, len(items))
	for index, raw := range items {
		itemPath := fmt.Sprintf("%s[%d]", path, index)
		item, ok := object(raw)
		if !ok {
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath, Message: "must be an object"})
			continue
		}
		typeName, _ := stringValue(item["type"])
		switch typeName {
		case "text":
			if item["cache_control"] != nil {
				*diagnostics = append(*diagnostics, Warning(itemPath+".cache_control", "prompt cache control is provider-specific"))
			}
			text, ok := stringValue(item["text"])
			if !ok {
				*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath + ".text", Message: "must be a string"})
				continue
			}
			contents = append(contents, Content{Type: ContentText, Text: text})
		case "tool_use":
			id, idOK := stringValue(item["id"])
			name, nameOK := stringValue(item["name"])
			arguments, argsOK := parseArguments(item["input"])
			if !idOK || !nameOK || !argsOK {
				*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath, Message: "tool_use requires string id/name and object input"})
				continue
			}
			contents = append(contents, Content{Type: ContentToolCall, ID: id, Name: name, Arguments: arguments})
		case "tool_result":
			id, idOK := stringValue(item["tool_use_id"])
			text, textOK := anthropicToolResultText(item["content"])
			if !idOK || !textOK {
				*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath, Message: "tool_result requires tool_use_id and text content"})
				continue
			}
			isError, _ := boolValue(item["is_error"])
			contents = append(contents, Content{Type: ContentToolResult, ToolCallID: id, Text: text, IsError: isError})
		case "thinking", "redacted_thinking":
			*diagnostics = append(*diagnostics, Warning(itemPath, "extended-thinking history is not portable"))
		default:
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath + ".type", Message: fmt.Sprintf("content type %q is not portable", typeName)})
		}
	}
	return contents
}

func anthropicToolResultText(value any) (string, bool) {
	if text, ok := stringValue(value); ok {
		return text, true
	}
	items, ok := array(value)
	if !ok {
		return "", false
	}
	var builder strings.Builder
	for _, raw := range items {
		item, ok := object(raw)
		if !ok || item["type"] != "text" {
			return "", false
		}
		text, ok := stringValue(item["text"])
		if !ok {
			return "", false
		}
		builder.WriteString(text)
	}
	return builder.String(), true
}

func decodeAnthropicTools(value any, diagnostics *[]Diagnostic) []Tool {
	if value == nil {
		return nil
	}
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: "$.tools", Message: "must be an array"})
		return nil
	}
	tools := make([]Tool, 0, len(items))
	for index, raw := range items {
		item, ok := object(raw)
		if !ok {
			*diagnostics = append(*diagnostics, Diagnostic{Path: fmt.Sprintf("$.tools[%d]", index), Message: "must be an object"})
			continue
		}
		name, nameOK := stringValue(item["name"])
		schema, schemaOK := object(item["input_schema"])
		if !nameOK || !schemaOK {
			*diagnostics = append(*diagnostics, Diagnostic{Path: fmt.Sprintf("$.tools[%d]", index), Message: "requires name and input_schema"})
			continue
		}
		description, _ := stringValue(item["description"])
		tools = append(tools, Tool{Name: name, Description: description, InputSchema: schema})
	}
	return tools
}

func decodeAnthropicToolChoice(value any, diagnostics *[]Diagnostic) *ToolChoice {
	if value == nil {
		return nil
	}
	choice, ok := object(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: "$.tool_choice", Message: "must be an object"})
		return nil
	}
	typeName, _ := stringValue(choice["type"])
	switch typeName {
	case "auto":
		return &ToolChoice{Type: "auto"}
	case "any":
		return &ToolChoice{Type: "required"}
	case "none":
		return &ToolChoice{Type: "none"}
	case "tool":
		name, ok := stringValue(choice["name"])
		if ok {
			return &ToolChoice{Type: "tool", Name: name}
		}
	}
	*diagnostics = append(*diagnostics, Diagnostic{Path: "$.tool_choice", Message: "contains an unsupported tool choice"})
	return nil
}

func decodeResponsesRequest(value map[string]any) (*Request, []Diagnostic, error) {
	request, diagnostics := baseRequest(value)
	diagnostics = append(diagnostics, diagnoseRequestKeys(value,
		[]string{"model", "input", "instructions", "tools", "tool_choice", "max_output_tokens", "temperature", "top_p", "stream"},
		map[string]string{
			"metadata": "metadata is not portable", "store": "response storage is provider-specific",
			"prompt_cache_key": "prompt cache routing is provider-specific", "client_metadata": "client metadata is not portable",
		})...)
	if maxTokens, ok := intValue(value["max_output_tokens"]); ok {
		request.MaxOutputTokens = maxTokens
	} else if value["max_output_tokens"] != nil {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.max_output_tokens", Message: "must be a positive integer"})
	}
	if instructions, ok := stringValue(value["instructions"]); ok && instructions != "" {
		request.Messages = append(request.Messages, Message{Role: "developer", Content: []Content{{Type: ContentText, Text: instructions}}})
	}
	request.Messages = append(request.Messages, decodeResponsesInput(value["input"], &diagnostics)...)
	request.Tools = decodeResponsesTools(value["tools"], &diagnostics)
	request.ToolChoice = decodeOpenAIToolChoice(value["tool_choice"], "$.tool_choice", &diagnostics)
	return request, diagnostics, nil
}

func decodeResponsesInput(value any, diagnostics *[]Diagnostic) []Message {
	if text, ok := stringValue(value); ok {
		return []Message{{Role: "user", Content: []Content{{Type: ContentText, Text: text}}}}
	}
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: "$.input", Message: "must be a string or item array"})
		return nil
	}
	messages := make([]Message, 0, len(items))
	for index, raw := range items {
		itemPath := fmt.Sprintf("$.input[%d]", index)
		item, ok := object(raw)
		if !ok {
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath, Message: "must be an object"})
			continue
		}
		typeName, _ := stringValue(item["type"])
		switch typeName {
		case "", "message":
			role, _ := stringValue(item["role"])
			if role != "system" && role != "developer" && role != "user" && role != "assistant" {
				*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath + ".role", Message: "contains an unsupported role"})
				continue
			}
			messages = append(messages, Message{Role: role, Content: decodeResponsesMessageContent(item["content"], itemPath+".content", diagnostics)})
		case "function_call":
			id, _ := stringValue(item["call_id"])
			if id == "" {
				id, _ = stringValue(item["id"])
			}
			name, nameOK := stringValue(item["name"])
			arguments, argsOK := parseArguments(item["arguments"])
			if id == "" || !nameOK || !argsOK {
				*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath, Message: "function_call requires call_id, name and JSON arguments"})
				continue
			}
			messages = append(messages, Message{Role: "assistant", Content: []Content{{Type: ContentToolCall, ID: id, Name: name, Arguments: arguments}}})
		case "function_call_output":
			id, idOK := stringValue(item["call_id"])
			if !idOK {
				*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath + ".call_id", Message: "must be a string"})
				continue
			}
			var output string
			if text, ok := stringValue(item["output"]); ok {
				output = text
			} else {
				encoded, _ := json.Marshal(item["output"])
				output = string(encoded)
			}
			messages = append(messages, Message{Role: "user", Content: []Content{{Type: ContentToolResult, ToolCallID: id, Text: output}}})
		case "reasoning":
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath, Message: "reasoning history is not portable without its provider state"})
		default:
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath + ".type", Message: fmt.Sprintf("input type %q is not portable", typeName)})
		}
	}
	return messages
}

func decodeResponsesMessageContent(value any, path string, diagnostics *[]Diagnostic) []Content {
	if text, ok := stringValue(value); ok {
		return []Content{{Type: ContentText, Text: text}}
	}
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: path, Message: "must be a string or content array"})
		return nil
	}
	var contents []Content
	for index, raw := range items {
		item, ok := object(raw)
		itemPath := fmt.Sprintf("%s[%d]", path, index)
		if !ok {
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath, Message: "must be an object"})
			continue
		}
		typeName, _ := stringValue(item["type"])
		if typeName != "input_text" && typeName != "output_text" && typeName != "text" {
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath + ".type", Message: fmt.Sprintf("content type %q is not portable", typeName)})
			continue
		}
		text, ok := stringValue(item["text"])
		if !ok {
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath + ".text", Message: "must be a string"})
			continue
		}
		contents = append(contents, Content{Type: ContentText, Text: text})
	}
	return contents
}

func decodeResponsesTools(value any, diagnostics *[]Diagnostic) []Tool {
	if value == nil {
		return nil
	}
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: "$.tools", Message: "must be an array"})
		return nil
	}
	var tools []Tool
	for index, raw := range items {
		item, ok := object(raw)
		path := fmt.Sprintf("$.tools[%d]", index)
		if !ok {
			*diagnostics = append(*diagnostics, Diagnostic{Path: path, Message: "must be an object"})
			continue
		}
		typeName, _ := stringValue(item["type"])
		if typeName != "function" {
			*diagnostics = append(*diagnostics, Diagnostic{Path: path + ".type", Message: fmt.Sprintf("tool type %q is not portable; only function tools are supported", typeName)})
			continue
		}
		name, nameOK := stringValue(item["name"])
		parameters, paramsOK := object(item["parameters"])
		if !nameOK || !paramsOK {
			*diagnostics = append(*diagnostics, Diagnostic{Path: path, Message: "function tool requires name and parameters"})
			continue
		}
		description, _ := stringValue(item["description"])
		strict, _ := boolValue(item["strict"])
		var strictPointer *bool
		if item["strict"] != nil {
			strictPointer = &strict
		}
		tools = append(tools, Tool{Name: name, Description: description, InputSchema: parameters, Strict: strictPointer})
	}
	return tools
}

func decodeChatRequest(value map[string]any) (*Request, []Diagnostic, error) {
	request, diagnostics := baseRequest(value)
	diagnostics = append(diagnostics, diagnoseRequestKeys(value,
		[]string{"model", "messages", "tools", "tool_choice", "max_tokens", "max_completion_tokens", "temperature", "top_p", "stop", "stream"},
		map[string]string{"stream_options": "stream usage options are transport-specific", "user": "user attribution is not portable"})...)
	if maxTokens, ok := intValue(value["max_completion_tokens"]); ok {
		request.MaxOutputTokens = maxTokens
	} else if maxTokens, ok := intValue(value["max_tokens"]); ok {
		request.MaxOutputTokens = maxTokens
	}
	if stops, ok := stringsValue(value["stop"]); ok {
		request.Stop = stops
	} else if value["stop"] != nil {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.stop", Message: "must be a string or string array"})
	}
	items, ok := array(value["messages"])
	if !ok {
		diagnostics = append(diagnostics, Diagnostic{Path: "$.messages", Message: "must be an array"})
	} else {
		for index, raw := range items {
			item, ok := object(raw)
			path := fmt.Sprintf("$.messages[%d]", index)
			if !ok {
				diagnostics = append(diagnostics, Diagnostic{Path: path, Message: "must be an object"})
				continue
			}
			role, _ := stringValue(item["role"])
			switch role {
			case "system", "developer", "user":
				request.Messages = append(request.Messages, Message{Role: role, Content: decodeChatTextContent(item["content"], path+".content", &diagnostics)})
			case "assistant":
				contents := decodeChatTextContent(item["content"], path+".content", &diagnostics)
				if calls, ok := array(item["tool_calls"]); ok {
					for callIndex, rawCall := range calls {
						call, _ := object(rawCall)
						fn, _ := object(call["function"])
						id, _ := stringValue(call["id"])
						name, _ := stringValue(fn["name"])
						arguments, argsOK := parseArguments(fn["arguments"])
						if id == "" || name == "" || !argsOK {
							diagnostics = append(diagnostics, Diagnostic{Path: fmt.Sprintf("%s.tool_calls[%d]", path, callIndex), Message: "invalid function tool call"})
							continue
						}
						contents = append(contents, Content{Type: ContentToolCall, ID: id, Name: name, Arguments: arguments})
					}
				}
				request.Messages = append(request.Messages, Message{Role: role, Content: contents})
			case "tool":
				id, _ := stringValue(item["tool_call_id"])
				text, _ := stringValue(item["content"])
				if id == "" {
					diagnostics = append(diagnostics, Diagnostic{Path: path + ".tool_call_id", Message: "must be a string"})
					continue
				}
				request.Messages = append(request.Messages, Message{Role: "user", Content: []Content{{Type: ContentToolResult, ToolCallID: id, Text: text}}})
			default:
				diagnostics = append(diagnostics, Diagnostic{Path: path + ".role", Message: "contains an unsupported role"})
			}
		}
	}
	request.Tools = decodeChatTools(value["tools"], &diagnostics)
	request.ToolChoice = decodeOpenAIToolChoice(value["tool_choice"], "$.tool_choice", &diagnostics)
	return request, diagnostics, nil
}

func decodeChatTextContent(value any, path string, diagnostics *[]Diagnostic) []Content {
	if value == nil {
		return nil
	}
	if text, ok := stringValue(value); ok {
		return []Content{{Type: ContentText, Text: text}}
	}
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: path, Message: "must be a string or text part array"})
		return nil
	}
	var contents []Content
	for index, raw := range items {
		item, ok := object(raw)
		itemPath := fmt.Sprintf("%s[%d]", path, index)
		if !ok || (item["type"] != "text" && item["type"] != "input_text") {
			*diagnostics = append(*diagnostics, Diagnostic{Path: itemPath, Message: "only text content is portable"})
			continue
		}
		text, ok := stringValue(item["text"])
		if ok {
			contents = append(contents, Content{Type: ContentText, Text: text})
		}
	}
	return contents
}

func decodeChatTools(value any, diagnostics *[]Diagnostic) []Tool {
	if value == nil {
		return nil
	}
	items, ok := array(value)
	if !ok {
		*diagnostics = append(*diagnostics, Diagnostic{Path: "$.tools", Message: "must be an array"})
		return nil
	}
	var tools []Tool
	for index, raw := range items {
		item, _ := object(raw)
		fn, _ := object(item["function"])
		name, _ := stringValue(fn["name"])
		parameters, ok := object(fn["parameters"])
		if item["type"] != "function" || name == "" || !ok {
			*diagnostics = append(*diagnostics, Diagnostic{Path: fmt.Sprintf("$.tools[%d]", index), Message: "invalid function tool"})
			continue
		}
		description, _ := stringValue(fn["description"])
		strict, strictOK := boolValue(fn["strict"])
		var strictPointer *bool
		if strictOK {
			strictPointer = &strict
		}
		tools = append(tools, Tool{Name: name, Description: description, InputSchema: parameters, Strict: strictPointer})
	}
	return tools
}

func decodeOpenAIToolChoice(value any, path string, diagnostics *[]Diagnostic) *ToolChoice {
	if value == nil {
		return nil
	}
	if typeName, ok := stringValue(value); ok {
		if typeName == "auto" || typeName == "none" || typeName == "required" {
			return &ToolChoice{Type: typeName}
		}
	}
	choice, ok := object(value)
	if ok && choice["type"] == "function" {
		if fn, nested := object(choice["function"]); nested {
			if name, valid := stringValue(fn["name"]); valid {
				return &ToolChoice{Type: "tool", Name: name}
			}
		}
		if name, valid := stringValue(choice["name"]); valid {
			return &ToolChoice{Type: "tool", Name: name}
		}
	}
	*diagnostics = append(*diagnostics, Diagnostic{Path: path, Message: "contains an unsupported tool choice"})
	return nil
}

func encodeAnthropicRequest(request *Request) (map[string]any, []Diagnostic) {
	value := map[string]any{
		"model":      request.Model,
		"max_tokens": max(request.MaxOutputTokens, 1024),
		"stream":     request.Stream,
	}
	var system []string
	var messages []map[string]any
	var diagnostics []Diagnostic
	appendMessage := func(role string, blocks []any) {
		if len(blocks) == 0 {
			return
		}
		if len(messages) > 0 && messages[len(messages)-1]["role"] == role {
			messages[len(messages)-1]["content"] = append(messages[len(messages)-1]["content"].([]any), blocks...)
			return
		}
		messages = append(messages, map[string]any{"role": role, "content": blocks})
	}
	for index, message := range request.Messages {
		if message.Role == "system" || message.Role == "developer" {
			for _, content := range message.Content {
				if content.Type != ContentText {
					diagnostics = append(diagnostics, Diagnostic{Path: fmt.Sprintf("$.messages[%d]", index), Message: "system messages may only contain text for Anthropic"})
					continue
				}
				system = append(system, content.Text)
			}
			continue
		}
		role := message.Role
		if role != "assistant" {
			role = "user"
		}
		var blocks []any
		for _, content := range message.Content {
			switch content.Type {
			case ContentText:
				blocks = append(blocks, map[string]any{"type": "text", "text": content.Text})
			case ContentToolCall:
				blocks = append(blocks, map[string]any{"type": "tool_use", "id": content.ID, "name": content.Name, "input": content.Arguments})
			case ContentToolResult:
				block := map[string]any{"type": "tool_result", "tool_use_id": content.ToolCallID, "content": content.Text}
				if content.IsError {
					block["is_error"] = true
				}
				blocks = append(blocks, block)
			}
		}
		appendMessage(role, blocks)
	}
	value["messages"] = messages
	if len(system) > 0 {
		value["system"] = strings.Join(system, "\n\n")
	}
	applyRequestOptions(value, request, "max_tokens")
	if len(request.Stop) > 0 {
		value["stop_sequences"] = request.Stop
	}
	if len(request.Tools) > 0 {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			tools = append(tools, map[string]any{"name": tool.Name, "description": tool.Description, "input_schema": tool.InputSchema})
		}
		value["tools"] = tools
	}
	if request.ToolChoice != nil {
		switch request.ToolChoice.Type {
		case "auto", "none":
			value["tool_choice"] = map[string]any{"type": request.ToolChoice.Type}
		case "required":
			value["tool_choice"] = map[string]any{"type": "any"}
		case "tool":
			value["tool_choice"] = map[string]any{"type": "tool", "name": request.ToolChoice.Name}
		}
	}
	return value, diagnostics
}

func encodeResponsesRequest(request *Request) (map[string]any, []Diagnostic) {
	value := map[string]any{"model": request.Model, "stream": request.Stream}
	var instructions []string
	var input []any
	for _, message := range request.Messages {
		if message.Role == "system" || message.Role == "developer" {
			instructions = append(instructions, contentText(message.Content))
			continue
		}
		var textParts []any
		for _, content := range message.Content {
			switch content.Type {
			case ContentText:
				partType := "input_text"
				if message.Role == "assistant" {
					partType = "output_text"
				}
				textParts = append(textParts, map[string]any{"type": partType, "text": content.Text})
			case ContentToolCall:
				arguments, _ := json.Marshal(content.Arguments)
				input = append(input, map[string]any{"type": "function_call", "call_id": content.ID, "name": content.Name, "arguments": string(arguments)})
			case ContentToolResult:
				input = append(input, map[string]any{"type": "function_call_output", "call_id": content.ToolCallID, "output": content.Text})
			}
		}
		if len(textParts) > 0 {
			input = append(input, map[string]any{"type": "message", "role": message.Role, "content": textParts})
		}
	}
	value["input"] = input
	if len(instructions) > 0 {
		value["instructions"] = strings.Join(instructions, "\n\n")
	}
	applyRequestOptions(value, request, "max_output_tokens")
	if len(request.Tools) > 0 {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			item := map[string]any{"type": "function", "name": tool.Name, "description": tool.Description, "parameters": tool.InputSchema}
			if tool.Strict != nil {
				item["strict"] = *tool.Strict
			}
			tools = append(tools, item)
		}
		value["tools"] = tools
	}
	if request.ToolChoice != nil {
		if request.ToolChoice.Type == "tool" {
			value["tool_choice"] = map[string]any{"type": "function", "name": request.ToolChoice.Name}
		} else {
			value["tool_choice"] = request.ToolChoice.Type
		}
	}
	return value, nil
}

func encodeChatRequest(request *Request) (map[string]any, []Diagnostic) {
	value := map[string]any{"model": request.Model, "stream": request.Stream}
	var messages []any
	for _, message := range request.Messages {
		var text strings.Builder
		var calls []any
		for _, content := range message.Content {
			switch content.Type {
			case ContentText:
				text.WriteString(content.Text)
			case ContentToolCall:
				arguments, _ := json.Marshal(content.Arguments)
				calls = append(calls, map[string]any{"id": content.ID, "type": "function", "function": map[string]any{"name": content.Name, "arguments": string(arguments)}})
			case ContentToolResult:
				messages = append(messages, map[string]any{"role": "tool", "tool_call_id": content.ToolCallID, "content": content.Text})
			}
		}
		if text.Len() > 0 || len(calls) > 0 || len(message.Content) == 0 {
			role := message.Role
			if role == "developer" {
				role = "system"
			}
			item := map[string]any{"role": role, "content": text.String()}
			if len(calls) > 0 {
				item["tool_calls"] = calls
				if text.Len() == 0 {
					item["content"] = nil
				}
			}
			messages = append(messages, item)
		}
	}
	value["messages"] = messages
	applyRequestOptions(value, request, "max_completion_tokens")
	if len(request.Stop) > 0 {
		value["stop"] = request.Stop
	}
	if len(request.Tools) > 0 {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			function := map[string]any{"name": tool.Name, "description": tool.Description, "parameters": tool.InputSchema}
			if tool.Strict != nil {
				function["strict"] = *tool.Strict
			}
			tools = append(tools, map[string]any{"type": "function", "function": function})
		}
		value["tools"] = tools
	}
	if request.ToolChoice != nil {
		if request.ToolChoice.Type == "tool" {
			value["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": request.ToolChoice.Name}}
		} else {
			value["tool_choice"] = request.ToolChoice.Type
		}
	}
	if request.Stream {
		value["stream_options"] = map[string]any{"include_usage": true}
	}
	return value, nil
}

func applyRequestOptions(value map[string]any, request *Request, maxKey string) {
	if request.MaxOutputTokens > 0 {
		value[maxKey] = request.MaxOutputTokens
	}
	if request.Temperature != nil {
		value["temperature"] = *request.Temperature
	}
	if request.TopP != nil {
		value["top_p"] = *request.TopP
	}
}

func diagnoseRequestKeys(value map[string]any, portable []string, warnings map[string]string) []Diagnostic {
	known := make(map[string]bool, len(portable)+len(warnings))
	for _, key := range portable {
		known[key] = true
	}
	for key := range warnings {
		known[key] = true
	}
	keys := make([]string, 0, len(value))
	for key := range value {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var diagnostics []Diagnostic
	for _, key := range keys {
		if message, ok := warnings[key]; ok {
			diagnostics = append(diagnostics, Warning("$."+key, message))
		} else if !known[key] {
			diagnostics = append(diagnostics, Diagnostic{Path: "$." + key, Message: "field is not supported for cross-protocol conversion"})
		}
	}
	return diagnostics
}
