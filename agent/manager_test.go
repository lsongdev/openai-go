package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lsongdev/miya-agents/acp"
	"github.com/lsongdev/miya-agents/config"
	"github.com/lsongdev/miya-agents/mcp"
	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/session"
)

func TestResolveAgentNameFallsBackWhenSessionProfileMissing(t *testing.T) {
	m := NewAgentManager(&config.Config{
		Profiles: map[string]*config.ProfileConfig{
			"coding": {Provider: "openai", ModelName: "gpt-4"},
		},
		Providers: map[string]*config.ProviderConfig{},
	})

	got, err := m.resolveAgentName("default")
	if err != nil {
		t.Fatalf("resolveAgentName: %v", err)
	}
	if got != "coding" {
		t.Fatalf("resolveAgentName = %q, want coding", got)
	}
}

func TestResolveAgentNamePrefersDefault(t *testing.T) {
	m := NewAgentManager(&config.Config{
		Profiles: map[string]*config.ProfileConfig{
			"default": {Provider: "openai", ModelName: "gpt-4"},
			"coding":  {Provider: "openai", ModelName: "gpt-4"},
		},
		Providers: map[string]*config.ProviderConfig{},
	})

	got, err := m.resolveAgentName("")
	if err != nil {
		t.Fatalf("resolveAgentName: %v", err)
	}
	if got != "default" {
		t.Fatalf("resolveAgentName = %q, want default", got)
	}
}

func TestResolveAgentNameReportsEmptyProfiles(t *testing.T) {
	m := NewAgentManager(&config.Config{
		Profiles:  map[string]*config.ProfileConfig{},
		Providers: map[string]*config.ProviderConfig{},
	})

	if _, err := m.resolveAgentName("default"); err == nil {
		t.Fatal("resolveAgentName succeeded, want error")
	}
}

func TestNewSessionUsesRequestedProfileAndListsMetadata(t *testing.T) {
	previousPath := config.ConfigPath
	config.ConfigPath = t.TempDir()
	t.Cleanup(func() { config.ConfigPath = previousPath })

	m := NewAgentManager(&config.Config{
		Profiles: map[string]*config.ProfileConfig{
			"default": {Provider: "openai", ModelName: "gpt-4"},
			"coding":  {Provider: "openai", ModelName: "gpt-5"},
		},
		Providers: map[string]*config.ProviderConfig{},
	})
	created, err := m.NewSession(context.Background(), &acp.NewSessionRequest{
		Meta: acp.Meta{acp.MiyaProfileMetaKey: "coding"},
	}, &recordingSender{})
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	if got := created.Meta[acp.MiyaProfileMetaKey]; got != "coding" {
		t.Fatalf("created profile = %v, want coding", got)
	}

	listed, err := m.ListSessions(context.Background(), &acp.ListSessionsRequest{})
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(listed.Sessions) != 1 {
		t.Fatalf("sessions = %d, want 1", len(listed.Sessions))
	}
	if got := listed.Sessions[0].Meta[acp.MiyaProfileMetaKey]; got != "coding" {
		t.Fatalf("listed profile = %v, want coding", got)
	}
}

func TestNewSessionRejectsUnknownRequestedProfile(t *testing.T) {
	m := NewAgentManager(&config.Config{
		Profiles: map[string]*config.ProfileConfig{
			"default": {Provider: "openai", ModelName: "gpt-4"},
		},
	})
	_, err := m.NewSession(context.Background(), &acp.NewSessionRequest{
		Meta: acp.Meta{acp.MiyaProfileMetaKey: "missing"},
	}, &recordingSender{})
	if err == nil {
		t.Fatal("NewSession succeeded, want unknown profile error")
	}
}

func TestUseAgentIncludesConfiguredMCPTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     int64  `json:"id"`
			Method string `json:"method"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		switch req.Method {
		case "initialize":
			writeMCPEvent(t, w, req.ID, map[string]any{
				"protocolVersion": "2025-03-26",
				"capabilities": map[string]any{
					"tools": map[string]any{},
				},
				"serverInfo": map[string]any{
					"name":    "coffee-test",
					"version": "1.0.0",
				},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeMCPEvent(t, w, req.ID, map[string]any{
				"tools": []map[string]any{{
					"name":        "queryShopList",
					"description": "query shops",
					"inputSchema": map[string]any{
						"type":       "object",
						"properties": map[string]any{},
					},
				}},
			})
		default:
			t.Fatalf("unexpected method: %s", req.Method)
		}
	}))
	defer server.Close()

	m := NewAgentManager(&config.Config{
		Profiles: map[string]*config.ProfileConfig{
			"default": {Provider: "openai", ModelName: "gpt-4"},
		},
		Providers: map[string]*config.ProviderConfig{
			"openai": {APIKey: "test-key"},
		},
		McpServers: map[string]*mcp.McpServerConfig{
			"coffee": {Type: "streamablehttp", URL: server.URL},
		},
	})

	ag, err := m.UseAgent("default")
	if err != nil {
		t.Fatalf("UseAgent: %v", err)
	}
	if _, ok := ag.tool("mcp_coffee_queryShopList"); !ok {
		t.Fatalf("missing MCP tool; tools = %#v", ag.tools)
	}
}

func TestUseAgentUsesAnthropicProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != "test-key" {
			t.Fatalf("x-api-key = %q", got)
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req["model"] != "claude-test" {
			t.Fatalf("model = %#v", req["model"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n")
		fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n")
		fmt.Fprint(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n")
		fmt.Fprint(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer server.Close()

	m := NewAgentManager(&config.Config{
		Profiles: map[string]*config.ProfileConfig{
			"default": {Provider: "claude", ModelName: "claude-test", Workspace: t.TempDir()},
		},
		Providers: map[string]*config.ProviderConfig{
			"claude": {Type: "anthropic", APIBase: server.URL, APIKey: "test-key"},
		},
	})
	ag, err := m.UseAgent("default")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := ag.Stream(context.Background(), &openai.ChatCompletionRequest{
		Model: "claude-test", Messages: []openai.ChatCompletionMessage{openai.UserMessage("hi")}, Stream: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	builder := openai.NewMessageBuilder()
	for chunk := range stream {
		if message := chunk.GetMessage(); message != nil {
			builder.Update(*message)
		}
	}
	if got := builder.Build().Content; got != "hello" {
		t.Fatalf("content = %q", got)
	}
}

func writeMCPEvent(t *testing.T, w http.ResponseWriter, id int64, result any) {
	t.Helper()
	w.Header().Set("Content-Type", "text/event-stream")
	payload, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"result":  result,
	})
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	_, _ = fmt.Fprintf(w, "event: message\ndata: %s\n\n", payload)
}

type recordingSender struct {
	updates []acp.SessionUpdate
}

func (s *recordingSender) Send(update acp.SessionUpdate) error {
	s.updates = append(s.updates, update)
	return nil
}

func TestReplaySessionReplaysEventLog(t *testing.T) {
	sess := &session.Session{
		Events: []session.Event{
			{ID: "evt_000001", Update: acp.SessionUpdate{
				SessionUpdate: "user_message_chunk",
				Content:       acp.ContentBlock{Type: "text", Text: "read file"},
			}},
			{ID: "evt_000002", Update: acp.SessionUpdate{
				SessionUpdate: "tool_call",
				ToolCall:      acpToolCall(ToolCallEvent{ID: "call-1", Name: "read_file", Arguments: `{"path":"README.md"}`}),
			}},
			{ID: "evt_000003", Update: acp.SessionUpdate{
				SessionUpdate:  "tool_call_update",
				ToolCallUpdate: acpToolCallUpdate(ToolCallEvent{ID: "call-1", Result: "hello"}, acp.ToolCallCompleted),
			}},
		},
	}
	sender := &recordingSender{}

	if err := replaySession(sess, sender); err != nil {
		t.Fatalf("replaySession: %v", err)
	}

	if len(sender.updates) != len(sess.Events) {
		t.Fatalf("updates = %d, want %d", len(sender.updates), len(sess.Events))
	}
	if sender.updates[0].SessionUpdate != "user_message_chunk" {
		t.Fatalf("first update = %q", sender.updates[0].SessionUpdate)
	}
	if sender.updates[1].ToolCall == nil || sender.updates[1].ToolCall.ToolCallID != "call-1" {
		t.Fatal("missing replayed tool_call update")
	}
	if sender.updates[2].ToolCallUpdate == nil || sender.updates[2].ToolCallUpdate.ToolCallID != "call-1" {
		t.Fatal("missing replayed tool_call_update update")
	}
}

func TestToolCallRawJSONValueMarshalsJSONSafely(t *testing.T) {
	call := acpToolCall(ToolCallEvent{ID: "call-raw", Name: "exec", Arguments: `{"command":"yt-dlp -x"}`})
	if !json.Valid(call.RawInput) {
		t.Fatalf("raw input is invalid JSON: %s", call.RawInput)
	}
	var raw map[string]any
	if err := json.Unmarshal(call.RawInput, &raw); err != nil {
		t.Fatalf("raw input unmarshal: %v", err)
	}
	if got := raw["command"]; got != "yt-dlp -x" {
		t.Fatalf("command = %#v", got)
	}

	update := toolCallDoneUpdate(ToolCallEvent{ID: "call-raw", Result: `bad \x escape from tool output`})
	if _, err := json.Marshal(update); err != nil {
		t.Fatalf("marshal update with raw output: %v", err)
	}
	if !json.Valid(update.ToolCallUpdate.RawOutput) {
		t.Fatalf("raw output is invalid JSON: %s", update.ToolCallUpdate.RawOutput)
	}
}
