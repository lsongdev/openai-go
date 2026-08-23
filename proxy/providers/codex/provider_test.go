package codex

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

func TestProviderFactory(t *testing.T) {
	auth := staticBearer{token: "token"}
	provider := Provider(auth, "gpt-test")
	if provider.Source != providers.SourceCodex || provider.NativeProtocol() != providers.ProtocolOpenAIResponses {
		t.Fatalf("provider source/protocol = %s/%s", provider.Source, provider.NativeProtocol())
	}
	if !provider.AlwaysStream || provider.Auth != auth {
		t.Fatalf("provider adapter not configured: %#v", provider)
	}
	body, err := provider.PrepareRequest(providers.ProtocolOpenAIResponses, []byte(`{"model":"gpt-test","stream":false,"store":true,"max_output_tokens":128}`))
	if err != nil {
		t.Fatal(err)
	}
	var request map[string]any
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	if request["stream"] != true || request["store"] != false || request["max_output_tokens"] != nil {
		t.Fatalf("prepared request = %#v", request)
	}
}

func TestStoreBearerTokenRefresh(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	store, err := NewStore(&Tokens{
		AccessToken:  "expired-token",
		RefreshToken: "refresh-1",
		AccountID:    "acct-9",
		ExpiresAt:    time.Now().Add(-time.Hour),
	}, path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Errorf("parse form: %v", err)
		}
		if r.FormValue("grant_type") != "refresh_token" || r.FormValue("refresh_token") != "refresh-1" {
			t.Errorf("form = %v", r.Form)
		}
		idToken := "header." + base64Payload(map[string]any{"chatgpt_account_id": "acct-9"}) + ".sig"
		_ = json.NewEncoder(w).Encode(tokenResponse{
			AccessToken: "fresh-token", RefreshToken: "refresh-2", IDToken: idToken, ExpiresIn: 3600,
		})
	}))
	defer upstream.Close()
	setTokenURL(t, upstream.URL)

	token, headers, err := store.BearerToken()
	if err != nil {
		t.Fatalf("BearerToken: %v", err)
	}
	if token != "fresh-token" || headers["chatgpt-account-id"] != "acct-9" {
		t.Fatalf("token/headers = %q/%v", token, headers)
	}
	reloaded, err := LoadStore(path)
	if err != nil {
		t.Fatalf("LoadStore: %v", err)
	}
	if got := reloaded.Tokens(); got.AccessToken != "fresh-token" || got.RefreshToken != "refresh-2" || got.AccountID != "acct-9" {
		t.Errorf("persisted tokens = %+v", got)
	}
}

func TestLoggedInRequiresOAuthAccessToken(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	dir := filepath.Join(home, ".codex")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "auth.json")
	if err := os.WriteFile(path, []byte(`{"OPENAI_API_KEY":"api-key-only"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if LoggedIn() {
		t.Fatal("API-key-only auth was treated as a Codex OAuth login")
	}
	if err := os.WriteFile(path, []byte(`{"tokens":{"access_token":"access-token","refresh_token":"refresh-token"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if !LoggedIn() {
		t.Fatal("OAuth access token was not detected")
	}
}

func TestExpiryOfJWT(t *testing.T) {
	want := time.Unix(1_900_000_000, 0)
	token := "header." + base64Payload(map[string]any{"exp": want.Unix()}) + ".signature"
	if got := expiryOf(token); !got.Equal(want) {
		t.Fatalf("expiry = %v, want %v", got, want)
	}
	if got := expiryOf("not-a-jwt"); !got.IsZero() {
		t.Fatalf("invalid token expiry = %v, want zero", got)
	}
}

type staticBearer struct{ token string }

func (s staticBearer) BearerToken() (string, map[string]string, error) {
	return s.token, nil, nil
}

func base64Payload(claims map[string]any) string {
	data, _ := json.Marshal(claims)
	return base64.RawURLEncoding.EncodeToString(data)
}

func setTokenURL(t *testing.T, url string) {
	t.Helper()
	old := oauthTokenURL
	oauthTokenURL = url
	t.Cleanup(func() { oauthTokenURL = old })
}
