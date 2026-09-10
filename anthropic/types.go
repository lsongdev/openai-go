package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Request is an Anthropic Messages API request.
type Request struct {
	Model         string    `json:"model"`
	MaxTokens     int       `json:"max_tokens"`
	Messages      []Message `json:"messages"`
	System        string    `json:"system,omitempty"`
	Tools         []Tool    `json:"tools,omitempty"`
	Stream        bool      `json:"stream,omitempty"`
	Temperature   *float64  `json:"temperature,omitempty"`
	TopP          *float64  `json:"top_p,omitempty"`
	StopSequences []string  `json:"stop_sequences,omitempty"`
}

func (r *Request) UnmarshalJSON(data []byte) error {
	type Alias Request
	aux := &struct {
		System json.RawMessage `json:"system,omitempty"`
		*Alias
	}{Alias: (*Alias)(r)}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	if len(aux.System) == 0 {
		return nil
	}
	var s string
	if err := json.Unmarshal(aux.System, &s); err == nil {
		r.System = s
		return nil
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	}
	if err := json.Unmarshal(aux.System, &blocks); err != nil {
		return fmt.Errorf("system must be a string or array of text blocks: %v", err)
	}
	var texts []string
	for _, b := range blocks {
		if b.Type == "text" {
			texts = append(texts, b.Text)
		}
	}
	r.System = strings.Join(texts, "\n")
	return nil
}

// Tool describes a client tool available to Claude.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// Message keeps the common text form terse while exposing Blocks for tool use
// and other structured Anthropic content. Set either Content or Blocks.
type Message struct {
	Role    string         `json:"role"`
	Content string         `json:"-"`
	Blocks  []ContentBlock `json:"-"`
}

func TextMessage(role, text string) Message {
	return Message{Role: role, Content: text}
}

func (m Message) MarshalJSON() ([]byte, error) {
	var content any = m.Content
	if m.Blocks != nil {
		content = m.Blocks
	}
	return json.Marshal(struct {
		Role    string `json:"role"`
		Content any    `json:"content"`
	}{m.Role, content})
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var wire struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	m.Role = wire.Role
	if err := json.Unmarshal(wire.Content, &m.Content); err == nil {
		m.Blocks = nil
		return nil
	}
	m.Content = ""
	if err := json.Unmarshal(wire.Content, &m.Blocks); err != nil {
		return fmt.Errorf("content must be a string or array of content blocks: %v", err)
	}
	return nil
}

func (m Message) contentBlocks() []ContentBlock {
	if m.Blocks != nil {
		return m.Blocks
	}
	if m.Content != "" {
		return []ContentBlock{{Type: "text", Text: m.Content}}
	}
	return nil
}

// Response is a non-streaming Anthropic Messages API response.
type Response struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"`
	Usage        Usage          `json:"usage"`
	StopReason   string         `json:"stop_reason,omitempty"`
	StopSequence string         `json:"stop_sequence,omitempty"`
}

// ContentBlock is a block in an Anthropic message or response.
type ContentBlock struct {
	Type      string          `json:"type"` // text, thinking, redacted_thinking, tool_use, tool_result
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

// Usage contains token usage from Anthropic.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// APIError represents an error returned by the Anthropic API.
type APIError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Event is a single SSE event in an Anthropic streaming response.
type Event struct {
	Type         string          `json:"type"`
	Index        *int            `json:"index,omitempty"`
	Message      *MessageStart   `json:"message,omitempty"`
	ContentBlock *ContentBlock   `json:"content_block,omitempty"`
	Delta        json.RawMessage `json:"delta,omitempty"`
	Usage        *Usage          `json:"usage,omitempty"`
	Error        *APIError       `json:"error,omitempty"`
}

// MessageStart is sent at the beginning of a streamed message.
type MessageStart struct {
	ID    string `json:"id"`
	Type  string `json:"type"`
	Role  string `json:"role"`
	Model string `json:"model"`
	Usage Usage  `json:"usage"`
}

// Delta represents the incremental payload inside streamed Anthropic events.
type Delta struct {
	Type        string `json:"type"` // text_delta, thinking_delta, input_json_delta, signature_delta
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}
