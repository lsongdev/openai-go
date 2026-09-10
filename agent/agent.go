package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/lsongdev/miya-agents/config"
	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/session"
	"github.com/lsongdev/miya-agents/tools"
)

// StreamFunc is the only model capability required by the agent loop.
type StreamFunc func(context.Context, *openai.ChatCompletionRequest) (<-chan openai.ChatCompletionResponse, error)

type Agent struct {
	Name   string
	Config *config.ProfileConfig
	Stream StreamFunc
	tools  []openai.Tool
}

func New(name string, cfg *config.ProfileConfig, stream StreamFunc) *Agent {
	return &Agent{Name: name, Config: cfg, Stream: stream}
}

// Use adds tools to the agent and returns it for chaining.
func (a *Agent) Use(tools ...openai.Tool) *Agent {
	a.tools = append(a.tools, tools...)
	return a
}

func (a *Agent) tool(name string) (openai.Tool, bool) {
	for _, tool := range a.tools {
		if tool.Def().Function.Name == name {
			return tool, true
		}
	}
	return nil, false
}

func (a *Agent) toolDefs() []openai.ToolDef {
	defs := make([]openai.ToolDef, len(a.tools))
	for i, tool := range a.tools {
		defs[i] = tool.Def()
	}
	return defs
}

func (a *Agent) RunAgentLoop(ctx context.Context, sess *session.Session, sink EventSink) error {
	if a.Stream == nil {
		return fmt.Errorf("agent has no model stream")
	}
	for {
		req := openai.ChatCompletionRequest{
			Model:    a.Config.ModelName,
			Messages: sess.Messages,
			Tools:    a.toolDefs(),
			Stream:   true,
		}
		resp, err := a.Stream(ctx, &req)
		if err != nil {
			return fmt.Errorf("create model stream: %w", err)
		}
		builder := openai.NewMessageBuilder()
		for chunk := range resp {
			if chunk.Error != nil {
				return fmt.Errorf("model stream: %s", chunk.Error.Message)
			}
			m := chunk.GetMessage()
			if m == nil {
				continue
			}
			builder.Update(*m)
			if m.ReasoningContent != "" {
				if err := sink.ThoughtDelta(m.ReasoningContent); err != nil {
					return err
				}
			}
			if m.Content != "" {
				if err := sink.AssistantDelta(m.Content); err != nil {
					return err
				}
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		respMessage := builder.Build()
		if respMessage.IsEmpty() {
			return fmt.Errorf("model stream closed without a response")
		}
		sess.AppendResponse(respMessage)
		if !respMessage.HasToolCall() {
			if err := sink.Usage(UsageEvent{}); err != nil {
				return err
			}
			a.AppendContextMaintenanceNotice(sess)
			if err := sess.Save(); err != nil {
				return fmt.Errorf("save session: %w", err)
			}
			if err := sink.Done(); err != nil {
				return err
			}
			return nil
		}

		for _, tc := range respMessage.ToolCalls {
			if tc.ID == "" {
				return fmt.Errorf("tool call %q is missing an id", tc.Function.Name)
			}
			tool, ok := a.tool(tc.Function.Name)
			if err := sink.ToolCallStart(ToolCallEvent{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
				Status:    "in_progress",
				Call:      tc,
			}); err != nil {
				return err
			}
			var result string
			status := "completed"
			if ok {
				result = tool.Run(ctx, tc.Function.Arguments)
				if tc.Function.Name == "attach_file" {
					attachmentResult, emitted, err := emitAttachedFileResult(sink, result)
					if err != nil {
						result = fmt.Sprintf("Error: %v", err)
					} else if emitted {
						result = attachmentResult
					}
				}
			} else {
				status = "failed"
				result = fmt.Sprintf("Error: unknown tool '%s'", tc.Function.Name)
			}
			if err := sink.ToolCallDone(ToolCallEvent{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
				Result:    result,
				Status:    status,
				Call:      tc,
			}); err != nil {
				return err
			}
			sess.Messages = append(sess.Messages, openai.ToolResultMessage(tc.ID, tc.Function.Name, result))
		}
		if err := sess.Save(); err != nil {
			return fmt.Errorf("save session: %w", err)
		}
	}
}

func emitAttachedFileResult(sink EventSink, result string) (string, bool, error) {
	var attachment tools.AttachFileResult
	if err := json.Unmarshal([]byte(result), &attachment); err != nil {
		return result, false, nil
	}
	if attachment.Type != tools.AttachFileEventType {
		return result, false, nil
	}
	if attachment.Data == "" && attachment.URI == "" {
		return "", false, fmt.Errorf("attachment has neither inline data nor URI")
	}
	if err := sink.AssistantFile(FileEvent{
		Name:     attachment.Name,
		MimeType: attachment.MimeType,
		Size:     attachment.Size,
		Data:     attachment.Data,
		URI:      attachment.URI,
	}); err != nil {
		return "", false, err
	}
	if attachment.Data != "" {
		return fmt.Sprintf("Attached %s (%s, %d bytes) inline.", attachment.Name, attachment.MimeType, attachment.Size), true, nil
	}
	return fmt.Sprintf("Attached %s (%s, %d bytes) as %s.", attachment.Name, attachment.MimeType, attachment.Size, attachment.URI), true, nil
}

func (a *Agent) NewSession() *session.Session {
	s := session.New(a.Name)
	prompt := a.readSystemPrompt()
	if prompt != "" {
		s.Messages = append(s.Messages, openai.SystemMessage(prompt))
	}
	return s
}

func (a *Agent) NewSessionWithPrompt(prompt string) *session.Session {
	s := session.New(a.Name)
	if prompt != "" {
		s.Messages = append(s.Messages, openai.SystemMessage(prompt))
	}
	return s
}

func (a *Agent) readSystemPrompt() string {
	workspace := a.Config.GetWorkspace()
	if workspace == "" {
		return "You are a helpful assistant."
	}
	data, err := os.ReadFile(filepath.Join(workspace, "AGENTS.md"))
	if err != nil {
		return "You are a helpful assistant."
	}
	return string(data)
}

func (a *Agent) BuildTools() {
	workspace := a.Config.GetWorkspace()
	if workspace != "" {
		_ = os.MkdirAll(workspace, 0755)
	}
	a.Use(
		&tools.WebFetchTool{},
		&tools.WebSearchTool{},
		&tools.ReadFileTool{Workspace: workspace},
		&tools.WriteFileTool{Workspace: workspace},
		&tools.AppendFileTool{Workspace: workspace},
		&tools.EditFileTool{Workspace: workspace},
		&tools.AttachFileTool{Workspace: workspace},
		&tools.ExecTool{
			Workspace:           workspace,
			DefaultTimeout:      tools.ExecDefaultTimeoutSeconds,
			RestrictToWorkspace: true,
		},
		&tools.SkillsTool{Workspace: filepath.Join(config.ConfigPath, "skills")},
	)
}
