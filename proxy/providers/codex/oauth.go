package codex

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
	"strings"
	"time"
)

const (
	oauthAuthorizeURL = "https://auth.openai.com/oauth/authorize"
	oauthClientID     = "app_EMoamEEZ73f0CkXaXp7hrann"
	oauthScope        = "openid profile email offline_access"
	callbackPort      = 1455
	callbackRedirect  = "http://localhost:1455/auth/callback"
)

// oauthTokenURL is a var so tests can point it at a stub server.
var oauthTokenURL = "https://auth.openai.com/oauth/token"

// Tokens are the ChatGPT OAuth tokens issued by auth.openai.com.
type Tokens struct {
	IDToken      string    `json:"id_token,omitempty"`
	AccessToken  string    `json:"access_token,omitempty"`
	RefreshToken string    `json:"refresh_token,omitempty"`
	AccountID    string    `json:"account_id,omitempty"`
	ExpiresAt    time.Time `json:"expires_at,omitempty"`
}

// Valid reports whether the access token is present and not about to expire.
func (t *Tokens) Valid() bool {
	return t != nil && t.AccessToken != "" &&
		(t.ExpiresAt.IsZero() || time.Until(t.ExpiresAt) > time.Minute)
}

// Login runs the full ChatGPT OAuth flow: it starts a local callback server,
// prints the authorization URL and exchanges the returned code for tokens.
func Login(ctx context.Context) (*Tokens, error) {
	state, verifier, challenge, err := newPKCE()
	if err != nil {
		return nil, err
	}

	codeCh := make(chan string, 1)
	mux := http.NewServeMux()
	mux.HandleFunc("/auth/callback", func(w http.ResponseWriter, r *http.Request) {
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

	authURL := fmt.Sprintf(
		"%s?response_type=code&client_id=%s&redirect_uri=%s&scope=%s&state=%s&code_challenge=%s&code_challenge_method=S256&id_token_add_organizations=true&codex_cli_simplified_flow=true&prompt=login",
		oauthAuthorizeURL,
		url.QueryEscape(oauthClientID),
		url.QueryEscape(callbackRedirect),
		url.QueryEscape(oauthScope),
		url.QueryEscape(state),
		url.QueryEscape(challenge),
	)
	fmt.Println("Open the following URL to log in to ChatGPT:")
	fmt.Println(authURL)

	var code string
	select {
	case err := <-listenErr:
		return nil, fmt.Errorf("codex: callback server: %w", err)
	case code = <-codeCh:
	case <-ctx.Done():
		srv.Shutdown(context.Background())
		return nil, ctx.Err()
	}
	srv.Shutdown(context.Background())

	if code == "" {
		return nil, errors.New("codex: no authorization code in callback")
	}
	return ExchangeCode(ctx, code, verifier)
}

// AuthorizeURL builds the authorization URL for manual (browser copy/paste)
// login. The returned verifier must be passed to ExchangeCode.
func AuthorizeURL() (authURL, state, verifier string, err error) {
	state, verifier, challenge, err := newPKCE()
	if err != nil {
		return "", "", "", err
	}
	authURL = fmt.Sprintf(
		"%s?response_type=code&client_id=%s&redirect_uri=%s&scope=%s&state=%s&code_challenge=%s&code_challenge_method=S256&id_token_add_organizations=true&codex_cli_simplified_flow=true&prompt=login",
		oauthAuthorizeURL,
		url.QueryEscape(oauthClientID),
		url.QueryEscape(callbackRedirect),
		url.QueryEscape(oauthScope),
		url.QueryEscape(state),
		url.QueryEscape(challenge),
	)
	return authURL, state, verifier, nil
}

// ExchangeCode exchanges an authorization code for OAuth tokens.
func ExchangeCode(ctx context.Context, code, verifier string) (*Tokens, error) {
	form := url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {callbackRedirect},
		"client_id":     {oauthClientID},
		"code_verifier": {verifier},
	}
	return requestTokens(ctx, form)
}

// Refresh exchanges a refresh token for a fresh set of tokens.
func Refresh(ctx context.Context, refreshToken string) (*Tokens, error) {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
		"client_id":     {oauthClientID},
	}
	return requestTokens(ctx, form)
}

type tokenResponse struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
}

func requestTokens(ctx context.Context, form url.Values) (*Tokens, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, oauthTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("codex: token request failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("codex: token request failed with %s: %s", resp.Status, string(body))
	}

	var tr tokenResponse
	if err := json.Unmarshal(body, &tr); err != nil {
		return nil, fmt.Errorf("codex: decode token response: %w", err)
	}

	tokens := &Tokens{
		IDToken:      tr.IDToken,
		AccessToken:  tr.AccessToken,
		RefreshToken: tr.RefreshToken,
	}
	if tr.ExpiresIn > 0 {
		tokens.ExpiresAt = time.Now().Add(time.Duration(tr.ExpiresIn) * time.Second)
	} else if exp := expiryOf(tr.AccessToken); !exp.IsZero() {
		tokens.ExpiresAt = exp
	}
	tokens.AccountID = accountIDOf(tokens)
	return tokens, nil
}

// accountIDOf extracts the ChatGPT account id from the id or access token.
func accountIDOf(tokens *Tokens) string {
	for _, token := range []string{tokens.IDToken, tokens.AccessToken} {
		if token == "" {
			continue
		}
		var claims struct {
			ChatGPTAccountID string `json:"chatgpt_account_id"`
		}
		if jwtClaims(token, &claims) == nil && claims.ChatGPTAccountID != "" {
			return claims.ChatGPTAccountID
		}
	}
	return ""
}

// jwtClaims decodes the payload of a JWT without verifying its signature.
func jwtClaims(token string, out any) error {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return errors.New("invalid JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return err
	}
	return json.Unmarshal(payload, out)
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
