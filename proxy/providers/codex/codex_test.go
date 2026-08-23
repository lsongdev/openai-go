package codex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/lsongdev/miya-agents/openai"
)

func TestNewRequestFromChatCompletion(t *testing.T) {
	req := &openai.ChatCompletionRequest{
		Model: "gpt-5-codex",
		Messages: []openai.ChatCompletionMessage{
			openai.SystemMessage("be brief"),
			openai.UserMessage("hi"),
			openai.AssistantMessage("hello"),
			openai.UserMessage("bye"),
		},
		MaxTokens: 100,
	}
	respReq := NewRequestFromChatCompletion(req)

	if respReq.Model != "gpt-5-codex" {
		t.Errorf("model = %q, want gpt-5-codex", respReq.Model)
	}
	if respReq.Instructions != "be brief" {
		t.Errorf("instructions = %q, want 'be brief'", respReq.Instructions)
	}
	if len(respReq.Input.Items) != 3 {
		t.Fatalf("input items = %d, want 3", len(respReq.Input.Items))
	}
	if respReq.Input.Items[0].Role != "user" || respReq.Input.Items[0].Content.Text != "hi" {
		t.Errorf("item[0] = %+v", respReq.Input.Items[0])
	}
	if respReq.Input.Items[1].Role != "assistant" || respReq.Input.Items[1].Content.Text != "hello" {
		t.Errorf("item[1] = %+v", respReq.Input.Items[1])
	}
	if respReq.MaxOutputTokens != 100 {
		t.Errorf("max_output_tokens = %d, want 100", respReq.MaxOutputTokens)
	}
}

func TestChatCompletionFromResponseObject(t *testing.T) {
	obj := &openai.ResponseObject{
		ID:        "resp_1",
		Object:    "response",
		CreatedAt: 123,
		Status:    "completed",
		Model:     "gpt-5-codex",
		Output: []openai.ResponseOutputItem{
			{Type: "reasoning", Summary: []openai.ResponseSummaryItem{{Type: "summary_text", Text: "thinking"}}},
			{Type: "message", Role: "assistant", Content: []openai.ResponseContentPart{
				{Type: "output_text", Text: "Hello"},
				{Type: "output_text", Text: " world"},
			}},
		},
		Usage: &openai.ResponseUsage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	chatResp := ChatCompletionFromResponseObject(obj)
	msg := chatResp.GetMessage()
	if msg == nil {
		t.Fatal("expected message")
	}
	if msg.Content != "Hello world" {
		t.Errorf("content = %q, want 'Hello world'", msg.Content)
	}
	if msg.ReasoningContent != "thinking" {
		t.Errorf("reasoning_content = %q, want 'thinking'", msg.ReasoningContent)
	}
	if chatResp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", chatResp.Choices[0].FinishReason)
	}
	if chatResp.Usage.PromptTokens != 10 || chatResp.Usage.CompletionTokens != 5 || chatResp.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", chatResp.Usage)
	}
}

func TestCreateResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "Bearer tok-123" {
			t.Errorf("authorization = %q", auth)
		}
		if acct := r.Header.Get("chatgpt-account-id"); acct != "acct-1" {
			t.Errorf("account id header = %q", acct)
		}
		var req openai.ResponseRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "gpt-5-codex" {
			t.Errorf("model = %q", req.Model)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openai.ResponseObject{
			ID: "resp_1", Object: "response", Status: "completed", Model: "gpt-5-codex",
			Output: []openai.ResponseOutputItem{{
				Type: "message", Role: "assistant",
				Content: []openai.ResponseContentPart{{Type: "output_text", Text: "Hi there"}},
			}},
			Usage: &openai.ResponseUsage{InputTokens: 3, OutputTokens: 2, TotalTokens: 5},
		})
	}))
	defer upstream.Close()

	client := &Client{BaseURL: upstream.URL, Tokens: staticTokens{token: "tok-123", accountID: "acct-1"}}
	obj, err := client.CreateResponse(context.Background(), &openai.ResponseRequest{Model: "gpt-5-codex"})
	if err != nil {
		t.Fatalf("CreateResponse: %v", err)
	}
	chatResp := ChatCompletionFromResponseObject(obj)
	if msg := chatResp.GetMessage(); msg == nil || msg.Content != "Hi there" {
		t.Errorf("unexpected message: %+v", chatResp)
	}
}

type staticTokens struct {
	token     string
	accountID string
}

func (s staticTokens) BearerToken() (string, map[string]string, error) {
	return s.token, map[string]string{"chatgpt-account-id": s.accountID}, nil
}

func TestCreateResponseStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		created, _ := json.Marshal(map[string]any{
			"type": "response.created",
			"response": map[string]any{
				"id": "resp_9", "object": "response", "created_at": 42, "model": "gpt-5-codex", "status": "in_progress",
			},
		})
		fmt.Fprintf(w, "event: response.created\ndata: %s\n\n", created)

		delta, _ := json.Marshal(map[string]any{"type": "response.output_text.delta", "delta": "Hel"})
		fmt.Fprintf(w, "event: response.output_text.delta\ndata: %s\n\n", delta)
		delta2, _ := json.Marshal(map[string]any{"type": "response.output_text.delta", "delta": "lo"})
		fmt.Fprintf(w, "event: response.output_text.delta\ndata: %s\n\n", delta2)

		done, _ := json.Marshal(map[string]any{
			"type": "response.completed",
			"response": map[string]any{
				"id": "resp_9", "object": "response", "created_at": 42, "model": "gpt-5-codex", "status": "completed",
				"usage": map[string]any{"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
			},
		})
		fmt.Fprintf(w, "event: response.completed\ndata: %s\n\n", done)
	}))
	defer upstream.Close()

	client := &Client{BaseURL: upstream.URL, Tokens: staticTokens{token: "tok"}}
	chunks, err := client.CreateResponseStream(context.Background(), &openai.ResponseRequest{Model: "gpt-5-codex"})
	if err != nil {
		t.Fatalf("CreateResponseStream: %v", err)
	}

	assembler := openai.NewResponseAssembler()
	count := 0
	for chunk := range chunks {
		assembler.Update(chunk)
		count++
	}
	if count < 3 {
		t.Fatalf("expected at least 3 chunks, got %d", count)
	}
	resp := assembler.Build()
	if msg := resp.GetMessage(); msg == nil || msg.Content != "Hello" {
		t.Errorf("assembled content = %+v, want 'Hello'", resp.GetMessage())
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 6 {
		t.Errorf("usage = %+v, want total 6", resp.Usage)
	}
}

func TestStoreBearerTokenRefresh(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/auth.json"

	store, err := NewStore(&Tokens{
		AccessToken:  "expired-token",
		RefreshToken: "refresh-1",
		AccountID:    "acct-9",
		ExpiresAt:    time.Now().Add(-time.Hour),
	}, path)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}

	// Point refresh at a local fake token endpoint.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Errorf("parse form: %v", err)
		}
		if r.FormValue("grant_type") != "refresh_token" || r.FormValue("refresh_token") != "refresh-1" {
			t.Errorf("form = %v", r.Form)
		}
		idToken := "header." + base64Payload(map[string]any{"chatgpt_account_id": "acct-9"}) + ".sig"
		json.NewEncoder(w).Encode(tokenResponse{
			AccessToken:  "fresh-token",
			RefreshToken: "refresh-2",
			IDToken:      idToken,
			ExpiresIn:    3600,
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
	if headers["chatgpt-account-id"] != "acct-9" {
		t.Errorf("headers = %v", headers)
	}

	// Reload from disk and confirm the refreshed tokens were persisted.
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
