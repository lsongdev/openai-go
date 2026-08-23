package openai

import (
	"net/http"
	"testing"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

func TestProvider(t *testing.T) {
	provider := Provider("openai", "https://api.openai.test", "first-key")
	if provider.Source != providers.SourceOpenAI {
		t.Fatalf("source = %q, want %q", provider.Source, providers.SourceOpenAI)
	}
	if provider.NativeProtocol() != providers.ProtocolOpenAIChat {
		t.Fatalf("protocol = %q, want %q", provider.NativeProtocol(), providers.ProtocolOpenAIChat)
	}

	provider.APIKey = "updated-key"
	request, err := http.NewRequest(http.MethodPost, provider.BaseURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := provider.Authenticate(request); err != nil {
		t.Fatal(err)
	}
	if got := request.Header.Get("Authorization"); got != "Bearer updated-key" {
		t.Fatalf("authorization = %q, want %q", got, "Bearer updated-key")
	}
}
