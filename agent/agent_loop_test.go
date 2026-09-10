package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lsongdev/miya-agents/config"
	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/session"
)

func fakeStream(chunks ...openai.ChatCompletionResponse) StreamFunc {
	return func(context.Context, *openai.ChatCompletionRequest) (<-chan openai.ChatCompletionResponse, error) {
		ch := make(chan openai.ChatCompletionResponse, len(chunks))
		for _, chunk := range chunks {
			ch <- chunk
		}
		close(ch)
		return ch, nil
	}
}

type discardSink struct{}

func (discardSink) AssistantDelta(string) error        { return nil }
func (discardSink) AssistantFile(FileEvent) error      { return nil }
func (discardSink) ThoughtDelta(string) error          { return nil }
func (discardSink) ToolCallStart(ToolCallEvent) error  { return nil }
func (discardSink) ToolCallDone(ToolCallEvent) error   { return nil }
func (discardSink) SessionInfo(SessionInfoEvent) error { return nil }
func (discardSink) Usage(UsageEvent) error             { return nil }
func (discardSink) Done() error                        { return nil }

func TestRunAgentLoopRejectsEmptyStream(t *testing.T) {
	ag := New("test", &config.ProfileConfig{ModelName: "test"}, fakeStream())

	err := ag.RunAgentLoop(context.Background(), session.New("test"), discardSink{})
	if err == nil || !strings.Contains(err.Error(), "closed without a response") {
		t.Fatalf("RunAgentLoop error = %v", err)
	}
}

func TestRunAgentLoopRejectsInterruptedStream(t *testing.T) {
	message := openai.ChatCompletionMessage{Role: openai.RoleAssistant, Content: "partial"}
	ag := New("test", &config.ProfileConfig{ModelName: "test"}, fakeStream(
		openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &message}}},
		openai.ChatCompletionResponse{Error: &openai.Error{Type: "stream_error", Message: "connection reset"}},
	))
	sess := session.New("test")

	err := ag.RunAgentLoop(context.Background(), sess, discardSink{})
	if err == nil || !strings.Contains(err.Error(), "connection reset") {
		t.Fatalf("RunAgentLoop error = %v", err)
	}
	if len(sess.Messages) != 0 {
		t.Fatalf("interrupted response was saved: %#v", sess.Messages)
	}
}

func TestRunAgentLoopRejectsToolCallWithoutID(t *testing.T) {
	message := openai.ChatCompletionMessage{
		Role: openai.RoleAssistant,
		ToolCalls: []openai.ToolCall{{
			Function: openai.FunctionCall{Name: "read_file", Arguments: `{}`},
		}},
	}
	ag := New("test", &config.ProfileConfig{ModelName: "test"}, fakeStream(
		openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &message}}},
	))

	err := ag.RunAgentLoop(context.Background(), session.New("test"), discardSink{})
	if err == nil || !strings.Contains(err.Error(), "missing an id") {
		t.Fatalf("RunAgentLoop error = %v", err)
	}
}

func TestRunAgentLoopReturnsSaveError(t *testing.T) {
	oldConfigPath := config.ConfigPath
	configPath := filepath.Join(t.TempDir(), "config-file")
	if err := os.WriteFile(configPath, []byte("not a directory"), 0600); err != nil {
		t.Fatal(err)
	}
	config.ConfigPath = configPath
	t.Cleanup(func() { config.ConfigPath = oldConfigPath })

	message := openai.ChatCompletionMessage{Role: openai.RoleAssistant, Content: "done"}
	ag := New("test", &config.ProfileConfig{ModelName: "test"}, fakeStream(
		openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Index: 0, Delta: &message}}},
	))

	err := ag.RunAgentLoop(context.Background(), session.New("test"), discardSink{})
	if err == nil || !strings.Contains(err.Error(), "save session") {
		t.Fatalf("RunAgentLoop error = %v", err)
	}
}
