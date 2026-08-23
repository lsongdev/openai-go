// Package codex adapts the ChatGPT Codex backend as a proxy provider source.
// The backend speaks OpenAI Responses; protocol handling lives in proxy/codec.
package codex

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

const (
	// DefaultBaseURL is the ChatGPT backend endpoint that serves Codex.
	DefaultBaseURL = "https://chatgpt.com/backend-api/codex"

	// DefaultModels are used when the Codex CLI model cache is unavailable.
	DefaultModels = "gpt-5-codex,gpt-5,gpt-5-mini"
)

// Provider returns a thin source adapter backed by ChatGPT OAuth credentials.
func Provider(tokens providers.BearerSource, models ...string) *providers.Provider {
	var catalog []map[string]any
	if len(models) == 0 {
		models, catalog = loadModelCatalog()
	}
	return &providers.Provider{
		Name:         "codex",
		Source:       providers.SourceCodex,
		Protocol:     providers.ProtocolOpenAIResponses,
		BaseURL:      DefaultBaseURL,
		Models:       models,
		ModelCatalog: catalog,
		Auth:         tokens,
		AlwaysStream: true,
		Authenticate: func(request *http.Request) error {
			token, headers, err := tokens.BearerToken()
			if err != nil {
				return err
			}
			request.Header.Set("Authorization", "Bearer "+token)
			request.Header.Set("OpenAI-Beta", "responses=experimental")
			for key, value := range headers {
				request.Header.Set(key, value)
			}
			return nil
		},
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
	models := make([]string, 0, strings.Count(DefaultModels, ",")+1)
	for _, model := range strings.Split(DefaultModels, ",") {
		models = append(models, strings.TrimSpace(model))
	}
	return models, nil
}

// expiryOf parses the exp claim of a JWT without verifying its signature.
func expiryOf(token string) time.Time {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
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
