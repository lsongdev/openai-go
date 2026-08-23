// Package claude implements login with a Claude.ai account via OAuth, so the
// proxy can serve Anthropic models with subscription credentials instead of
// API keys. Credentials are stored in ~/.claude/.credentials.json, compatible
// with the layout used by Claude Code.
package claude

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

const (
	oauthAuthorizeURL = "https://claude.ai/oauth/authorize"
	oauthClientID     = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
	oauthScope        = "org:create_api_key user:profile user:inference"
	callbackPort      = 54545
	callbackRedirect  = "http://localhost:54545/callback"

	// DefaultBaseURL is the Anthropic API endpoint used by the provider.
	DefaultBaseURL = "https://api.anthropic.com"

	// DefaultModels are registered when a Claude provider is added without
	// an explicit model list.
	DefaultModels = "claude-opus-4-1,claude-sonnet-4-5,claude-haiku-4-5"
)

// oauthTokenURL is a var so tests can point it at a stub server.
var oauthTokenURL = "https://console.anthropic.com/v1/oauth/token"

// Credentials are the OAuth tokens issued for a Claude.ai account.
type Credentials struct {
	AccessToken  string   `json:"accessToken"`
	RefreshToken string   `json:"refreshToken"`
	ExpiresAt    int64    `json:"expiresAt"` // unix milliseconds
	Scopes       []string `json:"scopes,omitempty"`
}

// Expired reports whether the access token is missing or about to expire.
func (c *Credentials) Expired() bool {
	return c == nil || c.AccessToken == "" ||
		(c.ExpiresAt != 0 && time.Until(time.UnixMilli(c.ExpiresAt)) < time.Minute)
}

// Login runs the full Claude OAuth flow: it starts a local callback server,
// prints the authorization URL and exchanges the returned code for tokens.
func Login(ctx context.Context) (*Credentials, error) {
	state, verifier, challenge, err := newPKCE()
	if err != nil {
		return nil, err
	}

	codeCh := make(chan string, 1)
	mux := http.NewServeMux()
	mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("state") != state {
			http.Error(w, "state mismatch", http.StatusBadRequest)
			return
		}
		fmt.Fprint(w, "<html><body><h3>Login successful.</h3>You can close this tab and return to the terminal.</body></html>")
		codeCh <- r.URL.Query().Get("code")
	})
	srv := &http.Server{Addr: fmt.Sprintf("127.0.0.1:%d", callbackPort), Handler: mux}
	listenErr := make(chan error, 1)
	go func() { listenErr <- srv.ListenAndServe() }()

	authURL := AuthorizeURL(state, challenge)
	fmt.Println("Open the following URL to log in to Claude:")
	fmt.Println(authURL)

	var code string
	select {
	case err := <-listenErr:
		return nil, fmt.Errorf("claude: callback server: %w", err)
	case code = <-codeCh:
	case <-ctx.Done():
		srv.Shutdown(context.Background())
		return nil, ctx.Err()
	}
	srv.Shutdown(context.Background())

	if code == "" {
		return nil, errors.New("claude: no authorization code in callback")
	}
	return ExchangeCode(ctx, code, verifier)
}

// AuthorizeURL builds the authorization URL for manual (browser copy/paste)
// login. The matching PKCE verifier must be passed to ExchangeCode; use
// NewPKCE to generate both.
func AuthorizeURL(state, challenge string) string {
	return fmt.Sprintf(
		"%s?response_type=code&client_id=%s&redirect_uri=%s&scope=%s&state=%s&code_challenge=%s&code_challenge_method=S256",
		oauthAuthorizeURL,
		url.QueryEscape(oauthClientID),
		url.QueryEscape(callbackRedirect),
		url.QueryEscape(oauthScope),
		url.QueryEscape(state),
		url.QueryEscape(challenge),
	)
}

// NewPKCE generates a state and PKCE pair for the manual flow.
func NewPKCE() (state, verifier, challenge string, err error) {
	return newPKCE()
}

// ExchangeCode exchanges an authorization code for credentials.
func ExchangeCode(ctx context.Context, code, verifier string) (*Credentials, error) {
	form := url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {callbackRedirect},
		"client_id":     {oauthClientID},
		"code_verifier": {verifier},
	}
	return requestTokens(ctx, form)
}

// Refresh exchanges a refresh token for fresh credentials.
func Refresh(ctx context.Context, refreshToken string) (*Credentials, error) {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
		"client_id":     {oauthClientID},
	}
	return requestTokens(ctx, form)
}

type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
	Scope        string `json:"scope"`
	Err          string `json:"error"`
	ErrDesc      string `json:"error_description"`
}

func requestTokens(ctx context.Context, form url.Values) (*Credentials, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, oauthTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("claude: token request failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("claude: token request failed with %s: %s", resp.Status, string(body))
	}

	var tr tokenResponse
	if err := json.Unmarshal(body, &tr); err != nil {
		return nil, fmt.Errorf("claude: decode token response: %w", err)
	}
	if tr.Err != "" {
		return nil, fmt.Errorf("claude: %s: %s", tr.Err, tr.ErrDesc)
	}

	creds := &Credentials{
		AccessToken:  tr.AccessToken,
		RefreshToken: tr.RefreshToken,
	}
	if tr.ExpiresIn > 0 {
		creds.ExpiresAt = time.Now().Add(time.Duration(tr.ExpiresIn) * time.Second).UnixMilli()
	}
	if tr.Scope != "" {
		creds.Scopes = strings.Fields(tr.Scope)
	}
	return creds, nil
}

// ---------------------------------------------------------------------------
// Local credential store
// ---------------------------------------------------------------------------

// Store persists Claude OAuth credentials at ~/.claude/.credentials.json,
// compatible with the layout used by Claude Code. It implements the
// providers.BearerSource interface.
type Store struct {
	path  string
	mu    sync.Mutex
	creds *Credentials
}

type credentialsJSON struct {
	ClaudeAiOauth *Credentials `json:"claudeAiOauth,omitempty"`
}

// DefaultStorePath returns ~/.claude/.credentials.json.
func DefaultStorePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".claude", ".credentials.json"), nil
}

// LoadStore loads credentials from path, defaulting to
// ~/.claude/.credentials.json.
func LoadStore(path ...string) (*Store, error) {
	p, err := storePath(path)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return nil, fmt.Errorf("claude: load credentials: %w", err)
	}
	var raw credentialsJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("claude: parse %s: %w", p, err)
	}
	if raw.ClaudeAiOauth == nil {
		return nil, fmt.Errorf("claude: no claudeAiOauth credentials in %s", p)
	}
	return &Store{path: p, creds: raw.ClaudeAiOauth}, nil
}

// NewStore creates an in-memory store persisted at path (defaulting to
// ~/.claude/.credentials.json).
func NewStore(creds *Credentials, path ...string) (*Store, error) {
	p, err := storePath(path)
	if err != nil {
		return nil, err
	}
	return &Store{path: p, creds: creds}, nil
}

func storePath(path []string) (string, error) {
	if len(path) > 0 && path[0] != "" {
		return path[0], nil
	}
	return DefaultStorePath()
}

// Path returns the file backing the store.
func (s *Store) Path() string {
	return s.path
}

// Credentials returns a copy of the currently held credentials.
func (s *Store) Credentials() *Credentials {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.creds == nil {
		return nil
	}
	copied := *s.creds
	return &copied
}

// SetCredentials replaces and persists the held credentials.
func (s *Store) SetCredentials(creds *Credentials) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.creds = creds
	return s.saveLocked()
}

// BearerToken implements providers.BearerSource: it returns a valid access
// token, transparently refreshing it when it is about to expire.
func (s *Store) BearerToken() (string, map[string]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.creds == nil || s.creds.AccessToken == "" {
		return "", nil, errors.New("claude: not logged in (no credentials at " + s.path + ")")
	}
	if s.creds.Expired() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		refreshed, err := Refresh(ctx, s.creds.RefreshToken)
		if err != nil {
			return "", nil, fmt.Errorf("claude: refresh token: %w", err)
		}
		if refreshed.RefreshToken == "" {
			refreshed.RefreshToken = s.creds.RefreshToken
		}
		s.creds = refreshed
		if err := s.saveLocked(); err != nil {
			return "", nil, err
		}
	}
	return s.creds.AccessToken, nil, nil
}

func (s *Store) saveLocked() error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(credentialsJSON{ClaudeAiOauth: s.creds}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0o600)
}

// LoggedIn reports whether local Claude credentials exist.
func LoggedIn() bool {
	_, err := LoadStore()
	return err == nil
}

// ---------------------------------------------------------------------------
// Provider registration
// ---------------------------------------------------------------------------

// Provider returns a proxy provider backed by the Anthropic API using the
// given OAuth credential store.
func Provider(store *Store, models ...string) *providers.Provider {
	if len(models) == 0 {
		models = defaultModelList()
	}
	return &providers.Provider{
		Name:    "claude",
		Type:    providers.ProviderTypeAnthropic,
		BaseURL: DefaultBaseURL,
		Models:  models,
		Auth:    store,
	}
}

func defaultModelList() []string {
	return strings.Split(DefaultModels, ",")
}

func newPKCE() (state, verifier, challenge string, err error) {
	verifierBytes := make([]byte, 32)
	if _, err = rand.Read(verifierBytes); err != nil {
		return "", "", "", err
	}
	verifier = base64.RawURLEncoding.EncodeToString(verifierBytes)

	stateBytes := make([]byte, 16)
	if _, err = rand.Read(stateBytes); err != nil {
		return "", "", "", err
	}
	state = base64.RawURLEncoding.EncodeToString(stateBytes)

	sum := sha256.Sum256([]byte(verifier))
	challenge = base64.RawURLEncoding.EncodeToString(sum[:])
	return state, verifier, challenge, nil
}
