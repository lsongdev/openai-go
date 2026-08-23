package claudecode

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

func TestStoreBearerTokenRefresh(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/.credentials.json"

	store, err := NewStore(&Credentials{
		AccessToken:  "expired-token",
		RefreshToken: "refresh-1",
		ExpiresAt:    time.Now().Add(-time.Hour).UnixMilli(),
	}, path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Errorf("parse form: %v", err)
		}
		if got := r.FormValue("grant_type"); got != "refresh_token" {
			t.Errorf("grant_type = %q", got)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "fresh-token",
			"refresh_token": "refresh-2",
			"expires_in":    3600,
			"scope":         "user:inference",
		})
	}))
	defer upstream.Close()
	setTokenURL(t, upstream.URL)

	token, headers, err := store.BearerToken()
	if err != nil {
		t.Fatalf("BearerToken: %v", err)
	}
	if token != "fresh-token" {
		t.Errorf("token = %q, want fresh-token", token)
	}
	if headers != nil {
		t.Errorf("headers = %v, want nil", headers)
	}

	reloaded, err := LoadStore(path)
	if err != nil {
		t.Fatalf("LoadStore: %v", err)
	}
	creds := reloaded.Credentials()
	if creds.AccessToken != "fresh-token" || creds.RefreshToken != "refresh-2" {
		t.Errorf("persisted credentials = %+v", creds)
	}
	if len(creds.Scopes) != 1 || creds.Scopes[0] != "user:inference" {
		t.Errorf("scopes = %v", creds.Scopes)
	}
}

func TestProviderFactory(t *testing.T) {
	store, err := NewStore(&Credentials{AccessToken: "tok", RefreshToken: "r", ExpiresAt: time.Now().Add(time.Hour).UnixMilli()}, t.TempDir()+"/c.json")
	if err != nil {
		t.Fatal(err)
	}
	p := Provider(store, "claude-sonnet-4-5")
	if p.Source != providers.SourceClaudeCode {
		t.Errorf("source = %s", p.Source)
	}
	if p.BaseURL != DefaultBaseURL {
		t.Errorf("base url = %s", p.BaseURL)
	}
	token, _, err := p.Auth.BearerToken()
	if err != nil || token != "tok" {
		t.Errorf("bearer token = %q, err = %v", token, err)
	}
}

func TestAuthorizeURL(t *testing.T) {
	state, verifier, challenge, err := NewPKCE()
	if err != nil {
		t.Fatal(err)
	}
	url := AuthorizeURL(state, challenge)
	for _, want := range []string{"claude.ai/oauth/authorize", "code_challenge_method=S256", challenge} {
		if !strings.Contains(url, want) {
			t.Errorf("url %q missing %q", url, want)
		}
	}
	if state == "" || verifier == "" {
		t.Error("expected non-empty state and verifier")
	}
}

func setTokenURL(t *testing.T, url string) {
	t.Helper()
	old := oauthTokenURL
	oauthTokenURL = url
	t.Cleanup(func() { oauthTokenURL = old })
}
