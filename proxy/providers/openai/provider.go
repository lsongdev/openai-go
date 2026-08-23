// Package openai adapts OpenAI-compatible API-key providers to the proxy.
// Protocol parsing and API clients remain in the top-level openai package.
package openai

import (
	"net/http"

	openaiapi "github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy/providers"
)

// Provider creates a thin OpenAI-compatible provider source.
func Provider(name, baseURL, apiKey string) *providers.Provider {
	provider := &providers.Provider{
		Name:     name,
		Source:   providers.SourceOpenAI,
		Protocol: providers.ProtocolOpenAIChat,
		BaseURL:  baseURL,
		APIKey:   apiKey,
	}
	provider.Authenticate = func(request *http.Request) error {
		request.Header.Set("Authorization", "Bearer "+provider.APIKey)
		return nil
	}
	return provider
}

// Client returns the shared top-level OpenAI client configured for this source.
func Client(provider *providers.Provider, httpClient *http.Client) *openaiapi.Client {
	client, _ := openaiapi.NewClient(&openaiapi.Configuration{
		API:    provider.BaseURL,
		APIKey: provider.APIKey,
	})
	client.SetHTTPClient(httpClient)
	return client
}
