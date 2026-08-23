package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/lsongdev/miya-agents/openai"
)

type Source string

type Protocol string

const (
	SourceOpenAI     Source = "openai"
	SourceAnthropic  Source = "anthropic"
	SourceCodex      Source = "codex"
	SourceClaudeCode Source = "claudecode"

	ProtocolOpenAIChat      Protocol = "openai.chat.v1"
	ProtocolOpenAIResponses Protocol = "openai.responses.v1"
	ProtocolAnthropic       Protocol = "anthropic.messages.v1"
)

// BearerSource supplies rotating OAuth credentials for a provider. When set on
// a Provider it takes precedence over the static APIKey.
type BearerSource interface {
	// BearerToken returns the current bearer token plus any extra headers that
	// must accompany requests authenticated with it.
	BearerToken() (token string, headers map[string]string, err error)
}

// Client is the raw transport surface implemented by the top-level provider
// clients. Responses remain untouched for protocol handlers to proxy or decode.
type Client interface {
	NewRequest(context.Context, string, string, io.Reader) (*http.Request, error)
	Do(*http.Request) (*http.Response, error)
	SetHTTPClient(*http.Client)
}

type Provider struct {
	Name             string
	Source           Source
	Protocol         Protocol
	BaseURL          string
	APIKey           string
	Headers          map[string]string
	DefaultMaxTokens int
	Models           []string
	ModelAliases     map[string]string
	ModelCatalog     []map[string]any
	Client           Client
	PrepareRequest   func(protocol Protocol, body []byte) ([]byte, error)
	AlwaysStream     bool

	// Auth exposes rotating OAuth credentials to source adapters.
	Auth BearerSource
}

func (p *Provider) NativeProtocol() Protocol {
	if p.Protocol != "" {
		return p.Protocol
	}
	switch p.Source {
	case SourceAnthropic, SourceClaudeCode:
		return ProtocolAnthropic
	case SourceCodex:
		return ProtocolOpenAIResponses
	default:
		return ProtocolOpenAIChat
	}
}

func (p *Provider) SupportsModel(model string) bool {
	if _, ok := p.ModelAliases[model]; ok {
		return true
	}
	for _, candidate := range p.Models {
		if candidate == model {
			return true
		}
	}
	return false
}

func (p *Provider) NewNativeRequest(ctx context.Context, protocol Protocol, body []byte) (*http.Request, error) {
	return p.newRequest(ctx, protocol, body, false)
}

func (p *Provider) NewTranslatedRequest(ctx context.Context, protocol Protocol, body []byte) (*http.Request, error) {
	return p.newRequest(ctx, protocol, body, true)
}

func (p *Provider) newRequest(ctx context.Context, protocol Protocol, body []byte, prepare bool) (*http.Request, error) {
	if p.NativeProtocol() != protocol {
		return nil, fmt.Errorf("provider %q uses %s, not %s", p.Name, p.NativeProtocol(), protocol)
	}
	endpoint, err := p.nativeEndpoint(protocol)
	if err != nil {
		return nil, err
	}
	body, err = p.rewriteModelAlias(body)
	if err != nil {
		return nil, fmt.Errorf("provider %q: rewrite model: %w", p.Name, err)
	}
	if prepare && p.PrepareRequest != nil {
		body, err = p.PrepareRequest(protocol, body)
		if err != nil {
			return nil, fmt.Errorf("provider %q: prepare request: %w", p.Name, err)
		}
	}
	var req *http.Request
	if p.Client != nil {
		req, err = p.Client.NewRequest(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	} else {
		req, err = http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	}
	if err != nil {
		return nil, fmt.Errorf("provider %q: create request: %w", p.Name, err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if p.Client == nil && p.Auth != nil {
		token, headers, err := p.Auth.BearerToken()
		if err != nil {
			return nil, fmt.Errorf("provider %q: authenticate request: %w", p.Name, err)
		}
		req.Header.Set("Authorization", "Bearer "+token)
		for key, value := range headers {
			req.Header.Set(key, value)
		}
	} else if p.Client == nil && p.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
	}
	for key, value := range p.Headers {
		req.Header.Set(key, value)
	}
	return req, nil
}

// SetHTTPClient binds the proxy's shared transport to the source client.
func (p *Provider) SetHTTPClient(client *http.Client) {
	if p.Client != nil {
		p.Client.SetHTTPClient(client)
	}
}

// Do executes an upstream request through the source client when available.
func (p *Provider) Do(fallback *http.Client, request *http.Request) (*http.Response, error) {
	if p.Client != nil {
		return p.Client.Do(request)
	}
	return fallback.Do(request)
}

func (p *Provider) rewriteModelAlias(body []byte) ([]byte, error) {
	if len(p.ModelAliases) == 0 {
		return body, nil
	}
	var request map[string]any
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, err
	}
	model, _ := request["model"].(string)
	upstream, ok := p.ModelAliases[model]
	if !ok || upstream == "" || upstream == model {
		return body, nil
	}
	request["model"] = upstream
	return json.Marshal(request)
}

func (p *Provider) nativeEndpoint(protocol Protocol) (string, error) {
	base, err := url.Parse(strings.TrimRight(p.BaseURL, "/"))
	if err != nil || base.Scheme == "" || base.Host == "" {
		return "", fmt.Errorf("provider %q: invalid base URL %q", p.Name, p.BaseURL)
	}
	var suffix string
	switch protocol {
	case ProtocolOpenAIChat:
		suffix = "chat/completions"
	case ProtocolOpenAIResponses:
		suffix = "responses"
	case ProtocolAnthropic:
		if path.Base(base.Path) == "v1" {
			suffix = "messages"
		} else {
			suffix = "v1/messages"
		}
	default:
		return "", fmt.Errorf("provider %q: unsupported native protocol %q", p.Name, protocol)
	}
	base.Path = path.Join(base.Path, suffix)
	return base.String(), nil
}

type RequestError struct {
	Status  int
	Message string
}

func (e *RequestError) Error() string {
	return e.Message
}

type RequestContext struct {
	RequestID   string
	Response    http.ResponseWriter
	Request     *http.Request
	Upstream    *Provider
	Input       *openai.ChatCompletionRequest
	RawInput    []byte
	InputFormat Protocol
	Diagnostics []string
}

// APIKey extracts the API key from the Authorization or x-api-key header.
// It returns an empty string if no valid key is found.
func (ctx *RequestContext) APIKey() string {
	auth := ctx.Request.Header.Get("Authorization")
	if auth == "" {
		auth = "Bearer " + ctx.Request.Header.Get("x-api-key")
	}
	return strings.TrimPrefix(auth, "Bearer ")
}

type ResponseContext struct {
	RequestID   string
	Response    http.ResponseWriter
	Request     *http.Request
	Input       *openai.ChatCompletionRequest
	Output      *openai.ChatCompletionResponse
	Error       error
	Duration    time.Duration
	Diagnostics []string
}
