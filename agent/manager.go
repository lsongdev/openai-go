package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/lsongdev/miya-agents/acp"
	"github.com/lsongdev/miya-agents/anthropic"
	"github.com/lsongdev/miya-agents/config"
	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/session"
	"github.com/lsongdev/miya-agents/tools"
)

type Manager struct {
	config   *config.Config
	sessions map[string]*session.Session
	mu       sync.RWMutex
}

func NewAgentManager(config *config.Config) *Manager {
	return &Manager{
		config:   config,
		sessions: make(map[string]*session.Session),
	}
}

func (m *Manager) UseAgent(name string) (*Agent, error) {
	profile, ok := m.config.Profiles[name]
	if !ok {
		return nil, fmt.Errorf("agent not found: %s", name)
	}
	provider, ok := m.config.Providers[profile.Provider]
	if !ok {
		return nil, fmt.Errorf("provider not found: %s", profile.Provider)
	}

	var stream StreamFunc
	switch strings.ToLower(strings.TrimSpace(provider.Type)) {
	case "", "openai":
		client, err := openai.NewClient(&openai.Configuration{API: provider.APIBase, APIKey: provider.APIKey})
		if err != nil {
			return nil, err
		}
		stream = client.CreateChatCompletionStream
	case "anthropic":
		client := anthropic.NewClient(&anthropic.Configuration{API: provider.APIBase, APIKey: provider.APIKey})
		stream = client.CreateChatCompletionStream
	default:
		return nil, fmt.Errorf("unsupported provider type %q", provider.Type)
	}

	a := New(name, profile, stream)
	a.BuildTools()
	mcpManager := tools.NewMcpManager(m.config.McpServers)
	for _, tool := range mcpManager.Tools {
		a.Use(tool)
	}
	a.Use(tools.NewSubagentTool(m))
	return a, nil
}

func (m *Manager) defaultAgentName() (string, error) {
	if _, ok := m.config.Profiles["default"]; ok {
		return "default", nil
	}
	names := make([]string, 0, len(m.config.Profiles))
	for name := range m.config.Profiles {
		names = append(names, name)
	}
	if len(names) == 0 {
		return "", fmt.Errorf("no profiles configured")
	}
	sort.Strings(names)
	return names[0], nil
}

func (m *Manager) resolveAgentName(name string) (string, error) {
	if name != "" {
		if _, ok := m.config.Profiles[name]; ok {
			return name, nil
		}
	}
	return m.defaultAgentName()
}

type captureWriter struct {
	sb strings.Builder
}

func (w *captureWriter) AssistantDelta(text string) error { w.sb.WriteString(text); return nil }
func (w *captureWriter) AssistantFile(event FileEvent) error {
	if event.URI != "" {
		w.sb.WriteString(event.URI)
		return nil
	}
	if event.Name != "" {
		w.sb.WriteString(event.Name)
	}
	return nil
}
func (w *captureWriter) ThoughtDelta(text string) error { return nil }
func (w *captureWriter) ToolCallStart(event ToolCallEvent) error {
	return nil
}
func (w *captureWriter) ToolCallDone(event ToolCallEvent) error {
	return nil
}
func (w *captureWriter) SessionInfo(event SessionInfoEvent) error { return nil }
func (w *captureWriter) Usage(event UsageEvent) error             { return nil }
func (w *captureWriter) Done() error                              { return nil }

func (m *Manager) RunAgent(ctx context.Context, name, prompt string) (string, error) {
	ag, err := m.UseAgent(name)
	if err != nil {
		return "", err
	}

	sess := ag.NewSession()
	sess.AppendRequest(prompt)
	RecordUserMessage(sess, prompt)
	if sess.Title == "" {
		sess.Title = sess.DisplayTitle()
	}

	writer := &captureWriter{}
	sink := NewRecordingSink(sess, writer)
	if sess.Title != "" {
		if err := sink.SessionInfo(SessionInfoEvent{Title: sess.Title}); err != nil {
			return "", err
		}
	}
	if err := sess.Save(); err != nil {
		return "", fmt.Errorf("save session: %w", err)
	}
	if err := ag.RunAgentLoop(ctx, sess, sink); err != nil {
		return "", fmt.Errorf("agent '%s' failed: %v", name, err)
	}

	return writer.sb.String(), nil
}

// acp.ServerHandler implementation

func (m *Manager) Initialize(ctx context.Context, req *acp.InitializeRequest) (*acp.InitializeResponse, error) {
	caps := acp.DefaultAgentCapabilities()
	caps.LoadSession = true
	caps.SessionCapabilities = acp.SessionCapabilities{
		List:   &acp.SessionListCapabilities{},
		Resume: &acp.SessionResumeCapabilities{},
		Close:  &acp.SessionCloseCapabilities{},
		Delete: &acp.SessionDeleteCapabilities{},
	}

	return &acp.InitializeResponse{
		ProtocolVersion:   req.ProtocolVersion,
		AgentCapabilities: caps,
		AgentInfo: &acp.Implementation{
			Name:    "miya",
			Version: "0.1.0",
		},
	}, nil
}

func (m *Manager) Authenticate(ctx context.Context, req *acp.AuthenticateRequest) (*acp.AuthenticateResponse, error) {
	return &acp.AuthenticateResponse{}, nil
}

func (m *Manager) NewSession(ctx context.Context, req *acp.NewSessionRequest, sender acp.SessionUpdateSender) (*acp.NewSessionResponse, error) {
	requestedProfile, _ := req.Meta[acp.MiyaProfileMetaKey].(string)
	requestedProfile = strings.TrimSpace(requestedProfile)
	var agentName string
	var err error
	if requestedProfile != "" {
		if _, ok := m.config.Profiles[requestedProfile]; !ok {
			return nil, fmt.Errorf("profile not found: %s", requestedProfile)
		}
		agentName = requestedProfile
	} else {
		agentName, err = m.defaultAgentName()
	}
	if err != nil {
		return nil, err
	}

	sess := session.New(agentName)
	if err := sess.Save(); err != nil {
		return nil, fmt.Errorf("save session: %w", err)
	}

	m.mu.Lock()
	m.sessions[sess.ID] = sess
	m.mu.Unlock()

	return &acp.NewSessionResponse{
		SessionID: acp.SessionID(sess.ID),
		Meta:      acp.Meta{acp.MiyaProfileMetaKey: agentName},
	}, nil
}

func (m *Manager) Prompt(ctx context.Context, req *acp.PromptRequest, sender acp.SessionUpdateSender) (*acp.PromptResponse, error) {
	sess, err := session.Load(string(req.SessionID))
	if err != nil {
		return nil, fmt.Errorf("load session: %w", err)
	}
	agentName, err := m.resolveAgentName(sess.AgentName)
	if err != nil {
		return nil, err
	}
	if sess.AgentName != agentName {
		sess.AgentName = agentName
	}

	var textParts []string
	for _, block := range req.Prompt {
		if block.Text != "" {
			textParts = append(textParts, block.Text)
		}
	}
	prompt := strings.Join(textParts, "\n")
	sess.AppendRequest(prompt)
	RecordUserMessage(sess, prompt)
	titleChanged := false
	if sess.Title == "" {
		sess.Title = sess.DisplayTitle()
		titleChanged = sess.Title != ""
	}

	ag, err := m.UseAgent(agentName)
	if err != nil {
		return nil, fmt.Errorf("use agent: %w", err)
	}

	sink := NewRecordingSink(sess, &acpSink{sessionID: req.SessionID, sender: sender})
	if titleChanged {
		if err := sink.SessionInfo(SessionInfoEvent{Title: sess.Title}); err != nil {
			return nil, err
		}
	}
	if err := sess.Save(); err != nil {
		return nil, fmt.Errorf("save session: %w", err)
	}
	if err := ag.RunAgentLoop(ctx, sess, sink); err != nil {
		return nil, fmt.Errorf("agent loop: %w", err)
	}

	return &acp.PromptResponse{StopReason: acp.StopEndTurn}, nil
}

type acpSink struct {
	sessionID acp.SessionID
	sender    acp.SessionUpdateSender
}

func (s *acpSink) AssistantDelta(text string) error {
	if text == "" {
		return nil
	}
	return s.sender.Send(assistantMessageUpdate(text))
}

func (s *acpSink) AssistantFile(event FileEvent) error {
	return s.sender.Send(fileMessageUpdate(event))
}

func (s *acpSink) ThoughtDelta(text string) error {
	if text == "" {
		return nil
	}
	return s.sender.Send(thoughtUpdate(text))
}

func (s *acpSink) ToolCallStart(event ToolCallEvent) error {
	return s.sender.Send(toolCallUpdate(event))
}

func (s *acpSink) ToolCallDone(event ToolCallEvent) error {
	return s.sender.Send(toolCallDoneUpdate(event))
}

func (s *acpSink) SessionInfo(event SessionInfoEvent) error {
	if event.Title == "" {
		return nil
	}
	return s.sender.Send(sessionInfoUpdate(event))
}

func (s *acpSink) Usage(event UsageEvent) error {
	return s.sender.Send(usageUpdate(event))
}

func (s *acpSink) Done() error {
	return nil
}

func (m *Manager) LoadSession(ctx context.Context, req *acp.LoadSessionRequest, sender acp.SessionUpdateSender) (*acp.LoadSessionResponse, error) {
	sess, err := session.Load(string(req.SessionID))
	if err != nil {
		return nil, fmt.Errorf("load session: %w", err)
	}
	if err := replaySession(sess, sender); err != nil {
		return nil, fmt.Errorf("replay session: %w", err)
	}
	return &acp.LoadSessionResponse{}, nil
}

func replaySession(sess *session.Session, sender acp.SessionUpdateSender) error {
	for _, event := range sess.Events {
		if err := sender.Send(event.Update); err != nil {
			return err
		}
	}
	return nil
}

func (m *Manager) ResumeSession(ctx context.Context, req *acp.ResumeSessionRequest) (*acp.ResumeSessionResponse, error) {
	if _, err := session.Load(string(req.SessionID)); err != nil {
		return nil, fmt.Errorf("load session: %w", err)
	}
	return &acp.ResumeSessionResponse{}, nil
}

func (m *Manager) CloseSession(ctx context.Context, req *acp.CloseSessionRequest) (*acp.CloseSessionResponse, error) {
	return &acp.CloseSessionResponse{}, nil
}

func (m *Manager) DeleteSession(ctx context.Context, req *acp.DeleteSessionRequest) (*acp.DeleteSessionResponse, error) {
	p := filepath.Join(config.ConfigPath, "sessions", string(req.SessionID)+".json")
	if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("delete session: %w", err)
	}
	return &acp.DeleteSessionResponse{}, nil
}

func (m *Manager) ListSessions(ctx context.Context, req *acp.ListSessionsRequest) (*acp.ListSessionsResponse, error) {
	sessions, err := session.List()
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	infos := make([]acp.SessionInfo, 0, len(sessions))
	for _, s := range sessions {
		var title *string
		if value := s.DisplayTitle(); value != "" {
			title = &value
		}
		var updatedAt *string
		if !s.UpdatedAt.IsZero() {
			value := s.UpdatedAt.Format(time.RFC3339)
			updatedAt = &value
		}
		infos = append(infos, acp.SessionInfo{
			SessionID: acp.SessionID(s.ID),
			Title:     title,
			UpdatedAt: updatedAt,
			Meta:      acp.Meta{acp.MiyaProfileMetaKey: s.AgentName},
		})
	}
	return &acp.ListSessionsResponse{Sessions: infos}, nil
}

func (m *Manager) SetSessionMode(ctx context.Context, req *acp.SetSessionModeRequest) (*acp.SetSessionModeResponse, error) {
	return &acp.SetSessionModeResponse{}, nil
}

func (m *Manager) SetSessionConfigOption(ctx context.Context, req *acp.SetSessionConfigOptionRequest) (*acp.SetSessionConfigOptionResponse, error) {
	return &acp.SetSessionConfigOptionResponse{}, nil
}

func (m *Manager) Logout(ctx context.Context, req *acp.LogoutRequest) (*acp.LogoutResponse, error) {
	return &acp.LogoutResponse{}, nil
}
