package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lsongdev/miya-agents/sse"
)

type Client struct {
	config  *Configuration
	client  *http.Client
	headers func() (map[string]string, error)
}

func NewClient(config *Configuration) (*Client, error) {
	return &Client{config: config, client: http.DefaultClient}, nil
}

func (c *Client) SetHTTPClient(client *http.Client) {
	c.client = client
}

// SetHeaders sets a per-request header provider. When set it replaces the
// default bearer authentication.
func (c *Client) SetHeaders(fn func() (map[string]string, error)) {
	c.headers = fn
}

// NewRequest builds an authenticated request without encoding or decoding its
// body. Endpoint may be absolute or relative to the configured API URL.
func (c *Client) NewRequest(ctx context.Context, method, endpoint string, body io.Reader) (*http.Request, error) {
	if !strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://") {
		endpoint = strings.TrimRight(c.config.API, "/") + "/" + strings.TrimLeft(endpoint, "/")
	}
	request, err := http.NewRequestWithContext(ctx, method, endpoint, body)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if c.headers != nil {
		headers, err := c.headers()
		if err != nil {
			return nil, fmt.Errorf("request headers: %w", err)
		}
		for key, value := range headers {
			request.Header.Set(key, value)
		}
	} else if c.config.APIKey != "" {
		request.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}
	return request, nil
}

// Do executes a request and returns the raw response. It does not interpret
// status codes or consume the response body; the caller owns response.Body.
func (c *Client) Do(request *http.Request) (*http.Response, error) {
	return c.client.Do(request)
}

func (client *Client) MakeRequest(ctx context.Context, path string, data any) (io.ReadCloser, error) {
	var body io.Reader
	if data != nil {
		payload, err := json.Marshal(data)
		if err != nil {
			return nil, fmt.Errorf("json error: %v", err)
		}
		body = bytes.NewReader(payload)
	}
	method := http.MethodGet
	if body != nil {
		method = http.MethodPost
	}
	request, err := client.NewRequest(ctx, method, path, body)
	if err != nil {
		return nil, err
	}
	res, err := client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("cannot make request: %v", err)
	}
	if res.StatusCode != http.StatusOK {
		res.Body.Close()
		return nil, fmt.Errorf("invalid status code: %s", res.Status)
	}
	return res.Body, nil
}

// Model represents a model in the API format.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

// Models fetches the list of available models from the API.
func (client *Client) Models() (models []Model, err error) {
	body, err := client.MakeRequest(context.Background(), "/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch models: %v", err)
	}
	defer body.Close()

	var response struct {
		Data []Model `json:"data"`
	}
	if err := json.NewDecoder(body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return response.Data, nil
}

func (resp *ChatCompletionResponse) GetFirstChoice() *ChatCompletionChoice {
	if len(resp.Choices) == 0 {
		return nil
	}
	return &resp.Choices[0]
}

func (resp *ChatCompletionResponse) GetMessage() *ChatCompletionMessage {
	choice := resp.GetFirstChoice()
	if choice == nil {
		return nil
	}
	if choice.Delta != nil && !choice.Delta.IsEmpty() {
		return choice.Delta
	}
	return choice.Message
}

type ChatCompletionChoice struct {
	Index   int                    `json:"index"`
	Message *ChatCompletionMessage `json:"message,omitzero"`
	Delta   *ChatCompletionMessage `json:"delta,omitzero"` // for streaming responses

	LogProbs     any    `json:"logprobs,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type ContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL *struct {
		URL    string `json:"url"`
		Detail string `json:"detail,omitempty"`
	} `json:"image_url,omitempty"`
}

type ChatCompletionMessage struct {
	Role             string `json:"role,omitempty"`              // system, user, assistant, tool
	Content          string `json:"content,omitempty"`           // text content (string or array of parts)
	ReasoningContent string `json:"reasoning_content,omitempty"` // deepseek-reasoner

	// tools calls request
	ToolCalls []ToolCall `json:"tool_calls,omitempty"` // for assistant messages
	// tools results
	ToolCallID string `json:"tool_call_id,omitempty"` // for tool result messages
	Name       string `json:"name,omitempty"`         // tool name for tool results
}

func (m *ChatCompletionMessage) UnmarshalJSON(data []byte) error {
	type Alias ChatCompletionMessage
	aux := &struct {
		Content json.RawMessage `json:"content,omitempty"`
		*Alias
	}{Alias: (*Alias)(m)}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	if len(aux.Content) == 0 {
		return nil
	}
	var s string
	if err := json.Unmarshal(aux.Content, &s); err == nil {
		m.Content = s
		return nil
	}
	var parts []ContentPart
	if err := json.Unmarshal(aux.Content, &parts); err != nil {
		return fmt.Errorf("content must be a string or array of content parts: %v", err)
	}
	var texts []string
	for _, p := range parts {
		if p.Type == "text" {
			texts = append(texts, p.Text)
		}
	}
	m.Content = strings.Join(texts, "\n")
	return nil
}

func (m *ChatCompletionMessage) IsEmpty() bool {
	return m.Role == "" && m.Content == "" && m.ReasoningContent == "" && len(m.ToolCalls) == 0
}

func (m *ChatCompletionMessage) HasToolCall() bool {
	return len(m.ToolCalls) > 0
}

// UserMessage creates a user message.
func UserMessage(content string) ChatCompletionMessage {
	return ChatCompletionMessage{Role: "user", Content: content}
}

// SystemMessage creates a system message.
func SystemMessage(content string) ChatCompletionMessage {
	return ChatCompletionMessage{Role: "system", Content: content}
}

// AssistantMessage creates an assistant message.
func AssistantMessage(content string) ChatCompletionMessage {
	return ChatCompletionMessage{Role: "assistant", Content: content}
}

// AssistantMessageWithTools creates an assistant message with tool calls.
func AssistantMessageWithTools(content string, toolCalls []ToolCall) ChatCompletionMessage {
	return ChatCompletionMessage{Role: "assistant", Content: content, ToolCalls: toolCalls}
}

// ToolResultMessage creates a tool result message.
func ToolResultMessage(toolCallID, name, content string) ChatCompletionMessage {
	return ChatCompletionMessage{Role: "tool", ToolCallID: toolCallID, Name: name, Content: content}
}

// CreateEmbeddings sends an embeddings request to the API.
func (c *Client) CreateEmbeddings(ctx context.Context, request *EmbeddingRequest) (*EmbeddingResponse, error) {
	body, err := c.MakeRequest(ctx, "/embeddings", request)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	data, err := io.ReadAll(body)
	if err != nil {
		return nil, err
	}
	var resp EmbeddingResponse
	err = json.Unmarshal(data, &resp)
	if err != nil {
		return nil, err
	}
	if resp.Error != nil && resp.Error.Code != "" {
		err = errors.New(resp.Error.Message)
	}
	return &resp, err
}

// CreateChatCompletion sends a non-streaming chat completion request.
func (c *Client) CreateChatCompletion(ctx context.Context, request *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	body, err := c.MakeRequest(ctx, "/chat/completions", request)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	data, err := io.ReadAll(body)
	if err != nil {
		return nil, err
	}
	var resp ChatCompletionResponse
	err = json.Unmarshal(data, &resp)
	if err != nil {
		return nil, err
	}
	if resp.Error != nil && resp.Error.Code != "" {
		err = errors.New(resp.Error.Message)
	}
	return &resp, err
}

// CreateChatCompletionStream sends a streaming chat completion request.
func (c *Client) CreateChatCompletionStream(ctx context.Context, request *ChatCompletionRequest) (<-chan ChatCompletionResponse, error) {
	resp := make(chan ChatCompletionResponse)
	streamRequest := *request
	streamRequest.Stream = true

	payload, err := json.Marshal(&streamRequest)
	if err != nil {
		return nil, fmt.Errorf("json error: %v", err)
	}

	req, err := c.NewRequest(ctx, http.MethodPost, "/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("invalid request: %v", err)
	}

	stream, err := sse.Do(ctx, c.client, req)
	if err != nil {
		return nil, err
	}

	go func() {
		defer close(resp)
		sendError := func(err error) {
			select {
			case resp <- ChatCompletionResponse{
				Error: &Error{Message: err.Error(), Type: "stream_error"},
			}:
			case <-ctx.Done():
			}
		}
		for {
			select {
			case <-ctx.Done():
				return
			case evt, ok := <-stream.Events:
				if !ok {
					select {
					case err, ok := <-stream.Err():
						if ok && err != nil {
							sendError(fmt.Errorf("sse stream: %w", err))
							return
						}
					default:
					}
					sendError(io.ErrUnexpectedEOF)
					return
				}
				if evt.Data == "[DONE]" {
					return
				}
				var chunk ChatCompletionResponse
				if err := json.Unmarshal([]byte(evt.Data), &chunk); err != nil {
					sendError(fmt.Errorf("decode stream event: %w", err))
					return
				}
				select {
				case resp <- chunk:
					continue
				case <-ctx.Done():
					return
				}
			case err, ok := <-stream.Err():
				if ok && err != nil {
					sendError(fmt.Errorf("sse stream: %w", err))
				} else {
					sendError(io.ErrUnexpectedEOF)
				}
				return
			}
		}
	}()
	return resp, nil
}
