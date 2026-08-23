package codec

import (
	"encoding/json"
	"fmt"
	"strings"
)

type Protocol string

const (
	OpenAIChat      Protocol = "openai.chat.v1"
	OpenAIResponses Protocol = "openai.responses.v1"
	Anthropic       Protocol = "anthropic.messages.v1"
)

type ContentType string

const (
	ContentText       ContentType = "text"
	ContentReasoning  ContentType = "reasoning"
	ContentToolCall   ContentType = "tool_call"
	ContentToolResult ContentType = "tool_result"
)

type Content struct {
	Type       ContentType
	Text       string
	ID         string
	Name       string
	Arguments  map[string]any
	ToolCallID string
	IsError    bool
}

type Message struct {
	Role    string
	Content []Content
}

type Tool struct {
	Name        string
	Description string
	InputSchema map[string]any
	Strict      *bool
}

type ToolChoice struct {
	Type string
	Name string
}

type Request struct {
	Model           string
	Messages        []Message
	Tools           []Tool
	ToolChoice      *ToolChoice
	MaxOutputTokens int
	Temperature     *float64
	TopP            *float64
	Stop            []string
	Stream          bool
}

type Usage struct {
	InputTokens           int
	OutputTokens          int
	TotalTokens           int
	CachedInputTokens     int
	CacheWriteInputTokens int
	ReasoningTokens       int
}

type Response struct {
	ID         string
	Model      string
	Content    []Content
	StopReason string
	Usage      *Usage
	CreatedAt  int64
}

type Diagnostic struct {
	Path     string `json:"path"`
	Message  string `json:"message"`
	Severity string `json:"severity,omitempty"`
}

type TranslationError struct {
	From        Protocol
	To          Protocol
	Diagnostics []Diagnostic
}

func (e *TranslationError) Error() string {
	parts := make([]string, 0, len(e.Diagnostics))
	for _, diagnostic := range e.Diagnostics {
		parts = append(parts, diagnostic.Path+": "+diagnostic.Message)
	}
	return fmt.Sprintf("cannot translate %s to %s: %s", e.From, e.To, strings.Join(parts, "; "))
}

func TranslateRequest(from, to Protocol, body []byte) ([]byte, *Request, []Diagnostic, error) {
	request, diagnostics, err := DecodeRequest(from, body)
	if err != nil {
		return nil, nil, nil, err
	}
	if blocking := BlockingDiagnostics(diagnostics); len(blocking) > 0 {
		return nil, nil, diagnostics, &TranslationError{From: from, To: to, Diagnostics: blocking}
	}
	encoded, encodedDiagnostics, err := EncodeRequest(to, request)
	if err != nil {
		return nil, nil, diagnostics, err
	}
	diagnostics = append(diagnostics, encodedDiagnostics...)
	if blocking := BlockingDiagnostics(diagnostics); len(blocking) > 0 {
		return nil, nil, diagnostics, &TranslationError{From: from, To: to, Diagnostics: blocking}
	}
	return encoded, request, diagnostics, nil
}

func TranslateResponse(from, to Protocol, body []byte) ([]byte, *Response, []Diagnostic, error) {
	response, diagnostics, err := DecodeResponse(from, body)
	if err != nil {
		return nil, nil, nil, err
	}
	if blocking := BlockingDiagnostics(diagnostics); len(blocking) > 0 {
		return nil, nil, diagnostics, &TranslationError{From: from, To: to, Diagnostics: blocking}
	}
	encoded, encodedDiagnostics, err := EncodeResponse(to, response)
	if err != nil {
		return nil, nil, diagnostics, err
	}
	diagnostics = append(diagnostics, encodedDiagnostics...)
	if blocking := BlockingDiagnostics(diagnostics); len(blocking) > 0 {
		return nil, nil, diagnostics, &TranslationError{From: from, To: to, Diagnostics: blocking}
	}
	return encoded, response, diagnostics, nil
}

func BlockingDiagnostics(diagnostics []Diagnostic) []Diagnostic {
	var blocking []Diagnostic
	for _, diagnostic := range diagnostics {
		if diagnostic.Severity != "warning" {
			blocking = append(blocking, diagnostic)
		}
	}
	return blocking
}

func Warning(path, message string) Diagnostic {
	return Diagnostic{Path: path, Message: message, Severity: "warning"}
}

func decodeObject(body []byte) (map[string]any, error) {
	decoder := json.NewDecoder(strings.NewReader(string(body)))
	decoder.UseNumber()
	var value map[string]any
	if err := decoder.Decode(&value); err != nil {
		return nil, fmt.Errorf("decode JSON: %w", err)
	}
	if value == nil {
		return nil, fmt.Errorf("request must be a JSON object")
	}
	return value, nil
}

func marshalObject(value map[string]any) ([]byte, error) {
	body, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode JSON: %w", err)
	}
	return body, nil
}

func object(value any) (map[string]any, bool) {
	result, ok := value.(map[string]any)
	return result, ok
}

func array(value any) ([]any, bool) {
	result, ok := value.([]any)
	return result, ok
}

func stringValue(value any) (string, bool) {
	result, ok := value.(string)
	return result, ok
}

func boolValue(value any) (bool, bool) {
	result, ok := value.(bool)
	return result, ok
}

func intValue(value any) (int, bool) {
	switch number := value.(type) {
	case json.Number:
		parsed, err := number.Int64()
		return int(parsed), err == nil
	case float64:
		return int(number), number == float64(int(number))
	case int:
		return number, true
	default:
		return 0, false
	}
}

func floatValue(value any) (*float64, bool) {
	var parsed float64
	switch number := value.(type) {
	case json.Number:
		value, err := number.Float64()
		if err != nil {
			return nil, false
		}
		parsed = value
	case float64:
		parsed = number
	default:
		return nil, false
	}
	return &parsed, true
}

func stringsValue(value any) ([]string, bool) {
	if text, ok := stringValue(value); ok {
		return []string{text}, true
	}
	items, ok := array(value)
	if !ok {
		return nil, false
	}
	result := make([]string, 0, len(items))
	for _, item := range items {
		text, ok := stringValue(item)
		if !ok {
			return nil, false
		}
		result = append(result, text)
	}
	return result, true
}

func parseArguments(value any) (map[string]any, bool) {
	if result, ok := object(value); ok {
		return result, true
	}
	text, ok := stringValue(value)
	if !ok {
		return nil, false
	}
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	var result map[string]any
	if err := decoder.Decode(&result); err != nil {
		return nil, false
	}
	return result, true
}

func contentText(contents []Content) string {
	var builder strings.Builder
	for _, content := range contents {
		if content.Type == ContentText {
			builder.WriteString(content.Text)
		}
	}
	return builder.String()
}
