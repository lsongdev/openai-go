package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/lsongdev/miya-agents/anthropic"
	"github.com/lsongdev/miya-agents/openai"
)

type ProviderType string

type Protocol string

const (
	ProviderTypeOpenAI    ProviderType = "openai"
	ProviderTypeAnthropic ProviderType = "anthropic"
	ProviderTypeCodex     ProviderType = "codex"

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

type Provider struct {
	Name             string
	Type             ProviderType
	Protocol         Protocol
	BaseURL          string
	APIKey           string
	Headers          map[string]string
	DefaultMaxTokens int
	Models           []string
	ModelAliases     map[string]string
	ModelCatalog     []map[string]any
	PrepareRequest   func(protocol Protocol, body []byte) ([]byte, error)
	AlwaysStream     bool

	// Auth supplies OAuth credentials (e.g. Codex or Claude login). When
	// non-nil it overrides APIKey for upstream authentication.
	Auth BearerSource
}

func (p *Provider) NativeProtocol() Protocol {
	if p.Protocol != "" {
		return p.Protocol
	}
	switch p.Type {
	case ProviderTypeAnthropic:
		return ProtocolAnthropic
	case ProviderTypeCodex:
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
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("provider %q: create request: %w", p.Name, err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if err := p.applyNativeAuth(req); err != nil {
		return nil, fmt.Errorf("provider %q: authenticate request: %w", p.Name, err)
	}
	for key, value := range p.Headers {
		req.Header.Set(key, value)
	}
	return req, nil
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

func (p *Provider) applyNativeAuth(req *http.Request) error {
	if p.Auth != nil {
		token, headers, err := p.Auth.BearerToken()
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+token)
		for key, value := range headers {
			req.Header.Set(key, value)
		}
	} else if p.Type == ProviderTypeAnthropic {
		req.Header.Set("x-api-key", p.APIKey)
	} else {
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
	}
	if p.Type == ProviderTypeAnthropic {
		req.Header.Set("anthropic-version", "2023-06-01")
		if p.Auth != nil {
			req.Header.Set("anthropic-beta", "oauth-2025-04-20")
		}
	}
	if p.Type == ProviderTypeCodex {
		req.Header.Set("OpenAI-Beta", "responses=experimental")
	}
	return nil
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

// NewOpenAIClient builds an OpenAI-compatible client for the provider.
func NewOpenAIClient(p *Provider, httpClient *http.Client) *openai.Client {
	client, _ := openai.NewClient(&openai.Configuration{
		API:    p.BaseURL,
		APIKey: p.APIKey,
	})
	client.SetHTTPClient(httpClient)
	return client
}

// NewAnthropicClient builds an Anthropic client for the provider. When the
// provider carries OAuth credentials (Auth), requests are authenticated with
// a bearer token and the OAuth beta header instead of x-api-key.
func NewAnthropicClient(p *Provider, httpClient *http.Client) *anthropic.Client {
	client := anthropic.NewClient(&anthropic.Configuration{
		API:    p.BaseURL,
		APIKey: p.APIKey,
	})
	client.SetHTTPClient(httpClient)
	if p.Auth != nil {
		client.SetHeaders(func() (map[string]string, error) {
			token, headers, err := p.Auth.BearerToken()
			if err != nil {
				return nil, err
			}
			hs := map[string]string{
				"Authorization":  "Bearer " + token,
				"anthropic-beta": "oauth-2025-04-20",
			}
			for k, v := range headers {
				hs[k] = v
			}
			return hs, nil
		})
	}
	return client
}
