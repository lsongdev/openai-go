// Package codex implements an upstream provider for the OpenAI Codex
// backend: the Responses API served behind ChatGPT, authenticated with
// ChatGPT OAuth tokens instead of API keys.
package codex

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy/providers"
	"github.com/lsongdev/miya-agents/sse"
)

const (
	// DefaultBaseURL is the ChatGPT backend endpoint that serves Codex.
	DefaultBaseURL = "https://chatgpt.com/backend-api/codex"

	// DefaultModels are registered when a Codex provider is added without
	// an explicit model list.
	DefaultModels = "gpt-5-codex,gpt-5,gpt-5-mini"
)

// TokenSource supplies OAuth tokens used to authenticate against the Codex
// backend.
type TokenSource interface {
	BearerToken() (token string, headers map[string]string, err error)
}

// Client talks to the OpenAI Codex backend, which speaks the Responses API.
type Client struct {
	BaseURL    string
	Tokens     TokenSource
	HTTPClient *http.Client
}

// NewClient creates a Codex client bound to a token source.
func NewClient(tokens TokenSource) *Client {
	return &Client{
		BaseURL:    DefaultBaseURL,
		Tokens:     tokens,
		HTTPClient: http.DefaultClient,
	}
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

func (c *Client) newRequest(ctx context.Context, req *openai.ResponseRequest) (*http.Request, error) {
	if c.Tokens == nil {
		return nil, fmt.Errorf("codex: no token source configured")
	}
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("codex: encode request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL()+"/responses", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("codex: create request: %w", err)
	}
	token, headers, err := c.Tokens.BearerToken()
	if err != nil {
		return nil, fmt.Errorf("codex: resolve token: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token)
	httpReq.Header.Set("OpenAI-Beta", "responses=experimental")
	for k, v := range headers {
		httpReq.Header.Set(k, v)
	}
	return httpReq, nil
}

func (c *Client) baseURL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return DefaultBaseURL
}

// CreateResponse sends a non-streaming Responses API request and returns the
// full response object.
func (c *Client) CreateResponse(ctx context.Context, req *openai.ResponseRequest) (*openai.ResponseObject, error) {
	httpReq, err := c.newRequest(ctx, req)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("codex: request failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("codex: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("codex: unexpected status %s: %s", resp.Status, string(body))
	}
	var obj openai.ResponseObject
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, fmt.Errorf("codex: decode response: %w", err)
	}
	if obj.Error != nil && obj.Error.Message != "" {
		return &obj, fmt.Errorf("codex: %s: %s", obj.Error.Code, obj.Error.Message)
	}
	return &obj, nil
}

// CreateResponseStream sends a streaming Responses API request and returns a
// channel of chat completion chunks assembled from the SSE events, so it can
// feed the shared protocol writers.
func (c *Client) CreateResponseStream(ctx context.Context, req *openai.ResponseRequest) (<-chan openai.ChatCompletionResponse, error) {
	req.Stream = true
	httpReq, err := c.newRequest(ctx, req)
	if err != nil {
		return nil, err
	}
	stream, err := sse.Do(ctx, c.httpClient(), httpReq)
	if err != nil {
		return nil, err
	}

	out := make(chan openai.ChatCompletionResponse)
	go func() {
		defer close(out)
		state := &streamState{}
		for {
			select {
			case <-ctx.Done():
				return
			case evt, ok := <-stream.Events:
				if !ok {
					return
				}
				state.handle(evt.Type, evt.Data, out)
			case <-stream.Err():
				return
			}
		}
	}()
	return out, nil
}

// streamState assembles chat completion chunks from Responses API events.
type streamState struct {
	id      string
	model   string
	created int64
	usage   *openai.ResponseUsage
}

func (s *streamState) chunk(delta *openai.ChatCompletionMessage, finishReason string) openai.ChatCompletionResponse {
	return openai.ChatCompletionResponse{
		ID:      s.id,
		Object:  "chat.completion.chunk",
		Created: s.created,
		Model:   s.model,
		Choices: []openai.ChatCompletionChoice{{
			Index:        0,
			Delta:        delta,
			FinishReason: finishReason,
		}},
	}
}

func (s *streamState) handle(eventType, data string, out chan<- openai.ChatCompletionResponse) {
	switch eventType {
	case "response.created", "response.in_progress":
		var evt struct {
			Response openai.ResponseObject `json:"response"`
		}
		if json.Unmarshal([]byte(data), &evt) != nil {
			return
		}
		s.id = evt.Response.ID
		s.model = evt.Response.Model
		s.created = evt.Response.CreatedAt
	case "response.output_text.delta":
		var evt struct {
			Delta string `json:"delta"`
		}
		if json.Unmarshal([]byte(data), &evt) != nil || evt.Delta == "" {
			return
		}
		out <- s.chunk(&openai.ChatCompletionMessage{Role: "assistant", Content: evt.Delta}, "")
	case "response.reasoning_summary_text.delta":
		var evt struct {
			Delta string `json:"delta"`
		}
		if json.Unmarshal([]byte(data), &evt) != nil || evt.Delta == "" {
			return
		}
		out <- s.chunk(&openai.ChatCompletionMessage{Role: "assistant", ReasoningContent: evt.Delta}, "")
	case "response.completed", "response.incomplete", "response.failed":
		var evt struct {
			Response openai.ResponseObject `json:"response"`
		}
		if json.Unmarshal([]byte(data), &evt) != nil {
			return
		}
		s.id = evt.Response.ID
		s.model = evt.Response.Model
		s.created = evt.Response.CreatedAt
		s.usage = evt.Response.Usage
		finishReason := "stop"
		if eventType == "response.incomplete" {
			finishReason = "length"
		} else if eventType == "response.failed" {
			finishReason = "error"
		}
		chunk := s.chunk(&openai.ChatCompletionMessage{}, finishReason)
		if s.usage != nil {
			chunk.Usage = &openai.CompletionUsage{
				PromptTokens:     s.usage.InputTokens,
				CompletionTokens: s.usage.OutputTokens,
				TotalTokens:      s.usage.TotalTokens,
			}
		}
		out <- chunk
	}
}

// ---------------------------------------------------------------------------
// Conversion: ChatCompletionRequest -> ResponseRequest
// ---------------------------------------------------------------------------

// NewRequestFromChatCompletion converts a ChatCompletionRequest into an
// equivalent Responses API request for the Codex backend.
func NewRequestFromChatCompletion(req *openai.ChatCompletionRequest) *openai.ResponseRequest {
	respReq := &openai.ResponseRequest{
		Model:           req.Model,
		MaxOutputTokens: req.MaxTokens,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		Stream:          req.Stream,
		User:            req.User,
	}

	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			if respReq.Instructions != "" {
				respReq.Instructions += "\n"
			}
			respReq.Instructions += m.Content
		case "assistant":
			respReq.Input.Items = append(respReq.Input.Items, openai.ResponseInputItem{
				Type:    "message",
				Role:    "assistant",
				Content: openai.ResponseInputContent{Text: m.Content},
			})
			for _, call := range m.ToolCalls {
				respReq.Input.Items = append(respReq.Input.Items, openai.ResponseInputItem{
					Type:      "function_call",
					CallID:    call.ID,
					Name:      call.Function.Name,
					Arguments: call.Function.Arguments,
				})
			}
		case "tool":
			respReq.Input.Items = append(respReq.Input.Items, openai.ResponseInputItem{
				Type:   "function_call_output",
				CallID: m.ToolCallID,
				Output: m.Content,
			})
		default:
			respReq.Input.Items = append(respReq.Input.Items, openai.ResponseInputItem{
				Type:    "message",
				Role:    "user",
				Content: openai.ResponseInputContent{Text: m.Content},
			})
		}
	}

	for _, t := range req.Tools {
		respReq.Tools = append(respReq.Tools, openai.ResponseTool{
			Type:        "function",
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}

	return respReq
}

// ---------------------------------------------------------------------------
// Conversion: ResponseObject -> ChatCompletionResponse
// ---------------------------------------------------------------------------

// ChatCompletionFromResponseObject converts a Responses API response object
// into an equivalent chat completion response.
func ChatCompletionFromResponseObject(obj *openai.ResponseObject) *openai.ChatCompletionResponse {
	msg := &openai.ChatCompletionMessage{Role: "assistant"}
	finishReason := "stop"

	for _, item := range obj.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					msg.Content += part.Text
				}
			}
		case "reasoning":
			for _, summary := range item.Summary {
				msg.ReasoningContent += summary.Text
			}
		case "function_call":
			msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: openai.FunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
			finishReason = "tool_calls"
		}
	}

	if obj.Status == "incomplete" {
		finishReason = "length"
	}

	chatResp := &openai.ChatCompletionResponse{
		ID:      obj.ID,
		Object:  "chat.completion",
		Created: obj.CreatedAt,
		Model:   obj.Model,
		Choices: []openai.ChatCompletionChoice{{
			Index:        0,
			Message:      msg,
			FinishReason: finishReason,
		}},
	}
	if obj.Usage != nil {
		chatResp.Usage = &openai.CompletionUsage{
			PromptTokens:     obj.Usage.InputTokens,
			CompletionTokens: obj.Usage.OutputTokens,
			TotalTokens:      obj.Usage.TotalTokens,
		}
	}
	return chatResp
}

// ---------------------------------------------------------------------------
// Provider registration
// ---------------------------------------------------------------------------

// Provider returns a proxy provider backed by the ChatGPT Codex endpoint.
func Provider(tokens providers.BearerSource, models ...string) *providers.Provider {
	var catalog []map[string]any
	if len(models) == 0 {
		models, catalog = loadModelCatalog()
	}
	return &providers.Provider{
		Name:         "codex",
		Type:         providers.ProviderTypeCodex,
		BaseURL:      DefaultBaseURL,
		Models:       models,
		ModelCatalog: catalog,
		Auth:         tokens,
		AlwaysStream: true,
		PrepareRequest: func(protocol providers.Protocol, body []byte) ([]byte, error) {
			if protocol != providers.ProtocolOpenAIResponses {
				return body, nil
			}
			var request map[string]any
			if err := json.Unmarshal(body, &request); err != nil {
				return nil, err
			}
			request["store"] = false
			request["stream"] = true
			delete(request, "max_output_tokens")
			return json.Marshal(request)
		},
	}
}

func defaultModelList() []string {
	models, _ := loadModelCatalog()
	return models
}

func loadModelCatalog() ([]string, []map[string]any) {
	if home, err := os.UserHomeDir(); err == nil {
		body, readErr := os.ReadFile(filepath.Join(home, ".codex", "models_cache.json"))
		if readErr == nil {
			var cache struct {
				Models []map[string]any `json:"models"`
			}
			if json.Unmarshal(body, &cache) == nil {
				models := make([]string, 0, len(cache.Models))
				for _, model := range cache.Models {
					if _, ok := model["supports_parallel_tool_calls"]; !ok {
						model["supports_parallel_tool_calls"] = true
					}
					if slug, ok := model["slug"].(string); ok && strings.TrimSpace(slug) != "" {
						models = append(models, slug)
					}
				}
				if len(models) > 0 {
					return models, cache.Models
				}
			}
		}
	}
	var models []string
	for _, m := range strings.Split(DefaultModels, ",") {
		models = append(models, strings.TrimSpace(m))
	}
	return models, nil
}

// expiryOf parses the exp claim of a JWT without verifying its signature.
func expiryOf(token string) time.Time {
	parts := strings.Split(token, ".")
	if len(parts) != 2 {
		return time.Time{}
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return time.Time{}
	}
	var claims struct {
		Exp int64 `json:"exp"`
	}
	if json.Unmarshal(payload, &claims) != nil || claims.Exp == 0 {
		return time.Time{}
	}
	return time.Unix(claims.Exp, 0)
}
