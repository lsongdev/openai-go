package anthropic

import (
	"context"
	"net/http"
	"testing"
)

func TestNewRequestAcceptsVersionedAPIBase(t *testing.T) {
	client := NewClient(&Configuration{API: "https://api.anthropic.com/v1", APIKey: "test-key"})
	request, err := client.NewRequest(context.Background(), http.MethodGet, "/v1/models", nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := request.URL.String(); got != "https://api.anthropic.com/v1/models" {
		t.Fatalf("URL = %q", got)
	}
}
