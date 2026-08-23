// Package anthropic adapts Anthropic API-key providers to the proxy.
// Protocol parsing and API clients remain in the top-level anthropic package.
package anthropic

import (
	"net/http"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

// Provider creates a thin Anthropic API provider source.
func Provider(name, baseURL, apiKey string) *providers.Provider {
	provider := &providers.Provider{
		Name:     name,
		Source:   providers.SourceAnthropic,
		Protocol: providers.ProtocolAnthropic,
		BaseURL:  baseURL,
		APIKey:   apiKey,
	}
	provider.Authenticate = func(request *http.Request) error {
		request.Header.Set("x-api-key", provider.APIKey)
		request.Header.Set("anthropic-version", "2023-06-01")
		return nil
	}
	return provider
}
