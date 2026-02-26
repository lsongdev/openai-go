package openai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type OpenAIErrorResponse struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    string `json:"code"`
}

type Configuration struct {
	API    string
	APIKey string `json:"api_key"`
}

type OpenAIClient struct {
	config *Configuration
	client *http.Client
}

func NewClient(config *Configuration) (openai *OpenAIClient, err error) {
	client := http.DefaultClient
	openai = &OpenAIClient{config, client}
	return
}

func (client OpenAIClient) MakeRequest(path string, data interface{}) (io.ReadCloser, error) {
	var req *http.Request
	var err error

	if data != nil {
		payload, err := json.Marshal(data)
		if err != nil {
			return nil, fmt.Errorf("json error: %v", err)
		}
		req, err = http.NewRequest("POST", client.config.API+path, bytes.NewBuffer(payload))
		if err != nil {
			return nil, fmt.Errorf("invalid request: %v", err)
		}
		req.Header.Add("Content-Type", "application/json")
	} else {
		req, err = http.NewRequest("GET", client.config.API+path, nil)
		if err != nil {
			return nil, fmt.Errorf("invalid request: %v", err)
		}
	}

	req.Header.Add("Authorization", "Bearer "+client.config.APIKey)
	res, err := client.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cannot make request: %v", err)
	}
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("invalid status code: %s", res.Status)
	}
	return res.Body, nil
}

// Model represents a model in the  API format
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

// Models fetches the list of available models from the API
func (client *OpenAIClient) Models() (models []Model, err error) {
	body, err := client.MakeRequest("/models", nil)
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

const (
	RoleSystem    = "system"
	RoleAssistant = "assistant"
	RoleUser      = "user"
	RoleTool      = "tool"
)

const (
	GPT3_5_Trubo      = "gpt-3.5-turbo"
	GPT3_5_Trubo_0301 = "gpt-3.5-turbo-0301"
	GPT4              = "gpt-4"
	GPT4o             = "gpt-4o"
)

// ToolCall represents a tool invocation by the model.
type ToolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call within a tool call.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// ToolDef defines a tool for the LLM (OpenAI function calling format).
type ToolDef struct {
	Type     string      `json:"type"` // "function"
	Function FunctionDef `json:"function"`
}

// FunctionDef defines a function that the model can call.
type FunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"` // JSON Schema
}

type ChatCompletionRequest struct {
	Model           string                  `json:"model,omitempty"`
	Messages        []ChatCompletionMessage `json:"messages,omitempty"`
	Tools           []ToolDef               `json:"tools,omitempty"`
	Temperature     float64                 `json:"temperature,omitempty"`
	TopP            string                  `json:"top_p,omitempty"`
	NumberOfChoices int                     `json:"n,omitempty"`
	Stream          bool                    `json:"stream,omitempty"`
	Stop            []string                `json:"stop,omitempty"`
	MaxTokens       int                     `json:"max_tokens,omitempty"`
	User            string                  `json:"user,omitempty"`
}

type CompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens,omitempty"`
	} `json:"prompt_tokens_details,omitempty"`

	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens,omitempty"` // deepseek-reasoner
	} `json:"completion_tokens_details,omitempty"`

	PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens,omitempty"`  // deepseek-reasoner
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens,omitempty"` // deepseek-reasoner
}

type ChatCompletionResponse struct {
	Error             OpenAIErrorResponse    `json:"error,omitempty"`
	ID                string                 `json:"id"`
	Object            string                 `json:"object"`
	Created           int64                  `json:"created"`
	Model             string                 `json:"model"`
	Choices           []ChatCompletionChoice `json:"choices"`
	Usage             CompletionUsage        `json:"usage,omitempty"`
	SystemFingerprint string                 `json:"system_fingerprint,omitempty"` // deepseek-reasoner
}

func (resp *ChatCompletionResponse) GetFirstChoice() *ChatCompletionChoice {
	for _, choice := range resp.Choices {
		return &choice
	}
	return nil
}

func (resp *ChatCompletionResponse) GetMessage() *ChatCompletionMessage {
	choice := resp.GetFirstChoice()
	if choice == nil {
		return nil
	}
	if !choice.Delta.IsEmpty() {
		return &choice.Delta
	}
	return &choice.Message
}

func (resp *ChatCompletionResponse) GetMessageContent() (content string, reasoning bool) {
	message := resp.GetMessage()
	if message.ReasoningContent != "" {
		return message.ReasoningContent, true
	}
	return message.Content, false
}

func (resp *ChatCompletionResponse) GetToolCalls() (toolCalls []ToolCall) {
	message := resp.GetMessage()
	if message == nil {
		return
	}
	for _, tool := range message.ToolCalls {
		if tool.Type != "function" {
			continue
		}
		toolCalls = append(toolCalls, tool)
	}
	return
}

func (resp *ChatCompletionResponse) HasToolCalls() bool {
	return len(resp.GetToolCalls()) > 0
}

type ChatCompletionChoice struct {
	Index   int                   `json:"index"`
	Message ChatCompletionMessage `json:"message,omitempty"`
	Delta   ChatCompletionMessage `json:"delta,omitempty"` // for streaming responses

	LogProbs     any    `json:"logprobs,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type ChatCompletionMessage struct {
	Role             string `json:"role"`                        // system, user, assistant, tool
	Content          string `json:"content,omitempty"`           // text content
	ReasoningContent string `json:"reasoning_content,omitempty"` // deepseek-reasoner

	// tools calls request
	ToolCalls []ToolCall `json:"tool_calls,omitempty"` // for assistant messages
	// tools results
	ToolCallID string `json:"tool_call_id,omitempty"` // for tool result messages
	Name       string `json:"name,omitempty"`         // tool name for tool results
}

func (m *ChatCompletionMessage) IsEmpty() bool {
	return m.Role == "" && m.Content == "" && m.ReasoningContent == "" && len(m.ToolCalls) == 0
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

// Create chat completion
// https://platform.openai.com/docs/api-reference/chat/create
func (openai *OpenAIClient) CreateChatCompletion(request *ChatCompletionRequest) (resp ChatCompletionResponse, err error) {
	data, err := openai.MakeRequest("/chat/completions", request)
	if err != nil {
		return
	}
	body, err := io.ReadAll(data)
	if err != nil {
		return resp, err
	}
	// log.Println(string(body))
	err = json.Unmarshal(body, &resp)
	if err != nil {
		return resp, err
	}
	if resp.Error.Code != "" {
		err = errors.New(resp.Error.Message)
	}
	return resp, err
}

// ChatCompletionStream represents a streaming response channel
type ChatCompletionStream struct {
	Error    chan error
	Response chan ChatCompletionResponse
}

// Close closes the stream channels
func (stream *ChatCompletionStream) Close() {
	close(stream.Error)
	close(stream.Response)
}

// CreateChatCompletionStream creates a streaming chat completion
func (openai *OpenAIClient) CreateChatCompletionStream(request *ChatCompletionRequest) (resp chan ChatCompletionResponse, err error) {
	resp = make(chan ChatCompletionResponse)
	data, err := openai.MakeRequest("/chat/completions", request)
	if err != nil {
		return
	}
	go func() {
		defer data.Close()
		defer close(resp)
		reader := bufio.NewReader(data)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				return
			}
			// Skip empty lines
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			// log.Println(line)
			// Remove "data: " prefix if present
			if after, ok := strings.CutPrefix(line, "data: "); ok {
				line = after
			}
			// Skip "[DONE]" message
			if line == "[DONE]" {
				return
			}
			// Parse the response
			var chunk ChatCompletionResponse
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				continue // Skip invalid JSON
			}
			// Send the chunk through the channel
			resp <- chunk
		}
	}()
	return resp, nil
}
