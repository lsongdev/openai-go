package codex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Store persists Codex OAuth credentials in ~/.codex/auth.json, compatible
// with the layout used by the OpenAI Codex CLI. It implements the
// providers.BearerSource interface.
type Store struct {
	path   string
	mu     sync.Mutex
	tokens *Tokens
}

// authJSON mirrors the on-disk format of ~/.codex/auth.json.
type authJSON struct {
	OPENAI_API_KEY string           `json:"OPENAI_API_KEY,omitempty"`
	Tokens         *persistedTokens `json:"tokens,omitempty"`
	LastRefresh    string           `json:"last_refresh,omitempty"`
}

type persistedTokens struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	AccountID    string `json:"account_id"`
}

// DefaultStorePath returns ~/.codex/auth.json.
func DefaultStorePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".codex", "auth.json"), nil
}

// LoadStore loads credentials from path, defaulting to ~/.codex/auth.json.
func LoadStore(path ...string) (*Store, error) {
	p, err := storePath(path)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return nil, fmt.Errorf("codex: load credentials: %w", err)
	}
	var raw authJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("codex: parse %s: %w", p, err)
	}
	store := &Store{path: p}
	if raw.Tokens != nil {
		tokens := &Tokens{
			IDToken:      raw.Tokens.IDToken,
			AccessToken:  raw.Tokens.AccessToken,
			RefreshToken: raw.Tokens.RefreshToken,
			AccountID:    raw.Tokens.AccountID,
		}
		if tokens.AccountID == "" {
			tokens.AccountID = accountIDOf(tokens)
		}
		if t, err := time.Parse(time.RFC3339, raw.LastRefresh); err == nil && !t.IsZero() {
			// Access tokens are valid for ~28 days; assume a conservative
			// expiry when no explicit timestamp is stored.
			tokens.ExpiresAt = t.Add(28 * 24 * time.Hour)
		}
		if exp := expiryOf(tokens.AccessToken); !exp.IsZero() {
			tokens.ExpiresAt = exp
		}
		store.tokens = tokens
	}
	return store, nil
}

// NewStore creates an in-memory store persisted at path (defaulting to
// ~/.codex/auth.json).
func NewStore(tokens *Tokens, path ...string) (*Store, error) {
	p, err := storePath(path)
	if err != nil {
		return nil, err
	}
	return &Store{path: p, tokens: tokens}, nil
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

// Tokens returns a copy of the currently held tokens.
func (s *Store) Tokens() *Tokens {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tokens == nil {
		return nil
	}
	copied := *s.tokens
	return &copied
}

// SetTokens replaces and persists the held tokens.
func (s *Store) SetTokens(tokens *Tokens) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tokens = tokens
	return s.saveLocked()
}

// BearerToken implements providers.BearerToken: it returns a valid access
// token, transparently refreshing it when it is about to expire.
func (s *Store) BearerToken() (string, map[string]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tokens == nil || s.tokens.AccessToken == "" {
		return "", nil, errors.New("codex: not logged in (no credentials at " + s.path + ")")
	}
	if !s.tokens.Valid() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		refreshed, err := Refresh(ctx, s.tokens.RefreshToken)
		if err != nil {
			return "", nil, fmt.Errorf("codex: refresh token: %w", err)
		}
		if refreshed.RefreshToken == "" {
			refreshed.RefreshToken = s.tokens.RefreshToken
		}
		if refreshed.AccountID == "" {
			refreshed.AccountID = s.tokens.AccountID
		}
		s.tokens = refreshed
		if err := s.saveLocked(); err != nil {
			return "", nil, err
		}
	}
	headers := map[string]string{}
	if s.tokens.AccountID != "" {
		headers["chatgpt-account-id"] = s.tokens.AccountID
	}
	return s.tokens.AccessToken, headers, nil
}

func (s *Store) saveLocked() error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	raw := authJSON{
		Tokens: &persistedTokens{
			IDToken:      s.tokens.IDToken,
			AccessToken:  s.tokens.AccessToken,
			RefreshToken: s.tokens.RefreshToken,
			AccountID:    s.tokens.AccountID,
		},
		LastRefresh: time.Now().Format(time.RFC3339),
	}
	data, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0o600)
}

// LoggedIn reports whether local Codex credentials exist.
func LoggedIn() bool {
	store, err := LoadStore()
	if err != nil {
		return false
	}
	tokens := store.Tokens()
	return tokens != nil && tokens.AccessToken != ""
}
