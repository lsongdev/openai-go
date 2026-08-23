package proxy

import (
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lsongdev/miya-agents/proxy/protocols"
	"github.com/lsongdev/miya-agents/proxy/providers"
)

// Type aliases re-exported for convenience of callers of this package.
type (
	Provider        = providers.Provider
	Source          = providers.Source
	Protocol        = providers.Protocol
	RequestContext  = providers.RequestContext
	ResponseContext = providers.ResponseContext
	RequestError    = providers.RequestError
)

const (
	SourceOpenAI            = providers.SourceOpenAI
	SourceAnthropic         = providers.SourceAnthropic
	SourceCodex             = providers.SourceCodex
	SourceClaudeCode        = providers.SourceClaudeCode
	ProtocolOpenAIChat      = providers.ProtocolOpenAIChat
	ProtocolOpenAIResponses = providers.ProtocolOpenAIResponses
	ProtocolAnthropic       = providers.ProtocolAnthropic
)

// Proxy is an LLM API proxy: it exposes OpenAI and Anthropic protocol
// endpoints and dispatches requests to the configured upstream providers.
type Proxy struct {
	mu         sync.RWMutex
	providers  map[string]*providers.Provider
	onRequest  func(*providers.RequestContext) error
	onResponse func(*providers.ResponseContext)
	client     *http.Client
	reqCounter uint64
}

func NewProxy() *Proxy {
	return &Proxy{
		providers: make(map[string]*providers.Provider),
		client: &http.Client{
			Timeout: 5 * time.Minute,
			Transport: &http.Transport{
				Proxy: http.ProxyFromEnvironment,
			},
		},
	}
}

func (p *Proxy) SetHTTPClient(client *http.Client) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.client = client
	for _, provider := range p.providers {
		provider.SetHTTPClient(client)
	}
}

func (p *Proxy) AddProvider(provider *providers.Provider) {
	p.mu.Lock()
	provider.SetHTTPClient(p.client)
	p.providers[provider.Name] = provider
	p.mu.Unlock()
}

func (p *Proxy) FindProviderForModel(model string) *providers.Provider {
	p.mu.RLock()
	defer p.mu.RUnlock()
	names := make([]string, 0, len(p.providers))
	for name := range p.providers {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if p.providers[name].SupportsModel(model) {
			return p.providers[name]
		}
	}
	return nil
}

func (p *Proxy) FindProvider(name string) *providers.Provider {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.providers[name]
}

func (p *Proxy) OnRequest(fn func(ctx *providers.RequestContext) error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.onRequest = fn
}

func (p *Proxy) OnResponse(fn func(ctx *providers.ResponseContext)) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.onResponse = fn
}

func (p *Proxy) nextRequestID() string {
	n := atomic.AddUint64(&p.reqCounter, 1)
	return fmt.Sprintf("req_%d_%d", time.Now().Unix(), n)
}

func (p *Proxy) listProviders() []*providers.Provider {
	p.mu.RLock()
	defer p.mu.RUnlock()
	names := make([]string, 0, len(p.providers))
	for name := range p.providers {
		names = append(names, name)
	}
	sort.Strings(names)
	list := make([]*providers.Provider, 0, len(names))
	for _, name := range names {
		list = append(list, p.providers[name])
	}
	return list
}

func (p *Proxy) env() *protocols.Env {
	p.mu.RLock()
	client := p.client
	onRequest := p.onRequest
	onResponse := p.onResponse
	p.mu.RUnlock()
	return &protocols.Env{
		FindProvider:  p.FindProviderForModel,
		ListProviders: p.listProviders,
		HTTPClient:    client,
		NextRequestID: p.nextRequestID,
		OnRequest:     onRequest,
		OnResponse:    onResponse,
	}
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	slog.Info("ServeHTTP", "method", req.Method, "url", req.URL)
	env := p.env()
	switch req.URL.Path {
	case "/v1/models":
		env.Models(w, req)
	case "/v1/messages":
		env.Messages(w, req)
	case "/v1/chat/completions":
		env.ChatCompletions(w, req)
	case "/v1/responses":
		env.Responses(w, req)
	case "/v1/embeddings":
		env.Embeddings(w, req)
	default:
		protocols.WriteError(w, http.StatusNotFound, "not found")
	}
}
