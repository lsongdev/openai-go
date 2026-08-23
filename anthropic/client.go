package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/lsongdev/miya-agents/sse"
)

// Configuration holds the API endpoint and key for the Anthropic client.
type Configuration struct {
	API    string
	APIKey string
}

// Client is a standard Anthropic Messages API client.
type Client struct {
	config *Configuration
	client *http.Client
	// headers optionally supplies per-request headers (e.g. OAuth bearer
	// authentication); when set it replaces the default x-api-key.
	headers func() (map[string]string, error)
}

type Model struct {
	ID          string `json:"id"`
	Type        string `json:"type,omitempty"`
	DisplayName string `json:"display_name,omitempty"`
}

// NewClient creates a new Anthropic client.
func NewClient(config *Configuration) *Client {
	return &Client{
		config: config,
		client: http.DefaultClient,
	}
}

// SetHTTPClient allows customizing the underlying HTTP client.
func (c *Client) SetHTTPClient(client *http.Client) {
	c.client = client
}

// SetHeaders sets a per-request header provider. When set it replaces the
// default x-api-key authentication.
func (c *Client) SetHeaders(fn func() (map[string]string, error)) {
	c.headers = fn
}

func (c *Client) applyHeaders(req *http.Request) error {
	if c.headers != nil {
		headers, err := c.headers()
		if err != nil {
			return err
		}
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		return nil
	}
	req.Header.Set("x-api-key", c.config.APIKey)
	return nil
}

func (c *Client) makeRequest(ctx context.Context, method, path string, body interface{}) (*http.Response, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("json marshal error: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.config.API+path, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("create request error: %w", err)
	}

	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if err := c.applyHeaders(req); err != nil {
		return nil, fmt.Errorf("auth error: %w", err)
	}
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API error %s: %s", resp.Status, string(respBody))
	}

	return resp, nil
}

func (c *Client) Models(ctx context.Context) ([]Model, error) {
	resp, err := c.makeRequest(ctx, "GET", "/v1/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch models: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Data []Model `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode models response: %w", err)
	}
	return result.Data, nil
}

// CreateMessage sends a non-streaming message request.
func (c *Client) CreateMessage(ctx context.Context, req *Request) (*Response, error) {
	resp, err := c.makeRequest(ctx, "POST", "/v1/messages", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var result Response
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// MessageStream represents a streaming response channel.
type MessageStream struct {
	Events chan Event
	Done   chan struct{}
}

// CreateMessageStream sends a streaming message request and returns a channel of SSE events.
func (c *Client) CreateMessageStream(ctx context.Context, req *Request) (*MessageStream, error) {
	req.Stream = true

	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("json marshal error: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.config.API+"/v1/messages", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create request error: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if err := c.applyHeaders(httpReq); err != nil {
		return nil, fmt.Errorf("auth error: %w", err)
	}
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	stream, err := sse.Do(ctx, c.client, httpReq)
	if err != nil {
		return nil, err
	}

	out := &MessageStream{
		Events: make(chan Event),
		Done:   make(chan struct{}),
	}

	go func() {
		defer close(out.Events)
		defer close(out.Done)

		for {
			select {
			case <-ctx.Done():
				return
			case evt, ok := <-stream.Events:
				if !ok {
					return
				}
				var event Event
				if err := json.Unmarshal([]byte(evt.Data), &event); err != nil {
					continue
				}
				event.Type = evt.Type
				out.Events <- event
			case <-stream.Err():
				return
			}
		}
	}()

	return out, nil
}
