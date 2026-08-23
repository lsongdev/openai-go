package anthropic

import (
	"context"
	"net/http"
	"testing"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

func TestProvider(t *testing.T) {
	provider := Provider("anthropic", "https://api.anthropic.test", "first-key")
	if provider.Source != providers.SourceAnthropic {
		t.Fatalf("source = %q, want %q", provider.Source, providers.SourceAnthropic)
	}
	if provider.NativeProtocol() != providers.ProtocolAnthropic {
		t.Fatalf("protocol = %q, want %q", provider.NativeProtocol(), providers.ProtocolAnthropic)
	}

	provider.APIKey = "updated-key"
	request, err := provider.Client.NewRequest(context.Background(), http.MethodPost, provider.BaseURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := request.Header.Get("x-api-key"); got != "updated-key" {
		t.Fatalf("x-api-key = %q, want %q", got, "updated-key")
	}
	if got := request.Header.Get("anthropic-version"); got != "2023-06-01" {
		t.Fatalf("anthropic-version = %q, want %q", got, "2023-06-01")
	}
}
