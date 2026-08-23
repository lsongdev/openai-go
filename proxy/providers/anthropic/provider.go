// Package anthropic adapts Anthropic API-key providers to the proxy.
// Protocol parsing and API clients remain in the top-level anthropic package.
package anthropic

import (
	anthropicapi "github.com/lsongdev/miya-agents/anthropic"
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
	client := anthropicapi.NewClient(&anthropicapi.Configuration{API: baseURL, APIKey: apiKey})
	client.SetHeaders(func() (map[string]string, error) {
		return map[string]string{"x-api-key": provider.APIKey}, nil
	})
	provider.Client = client
	return provider
}
