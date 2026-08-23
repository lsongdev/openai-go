package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/lsongdev/miya-agents/mcp"
)

type McpServerConfig = mcp.McpServerConfig

// Config is the root configuration structure.
type Config struct {
	Agents     []ACPAgentConfig            `json:"agents,omitempty" yaml:"agents,omitempty"`
	Profiles   map[string]*ProfileConfig   `json:"profiles" yaml:"profiles"`
	Providers  map[string]*ProviderConfig  `json:"providers" yaml:"providers"` // Provider configurations
	McpServers map[string]*McpServerConfig `json:"mcpServers,omitempty"`
	Tools      map[string]json.RawMessage  `json:"tools,omitempty" yaml:"tools,omitempty"`
	Logging    LoggingConfig               `json:"logging,omitempty" yaml:"logging,omitempty"`
}

// ACPAgentConfig contains an externally callable ACP agent endpoint.
type ACPAgentConfig struct {
	ID      string            `json:"id" yaml:"id"`
	Name    string            `json:"name,omitempty" yaml:"name,omitempty"`
	Enabled *bool             `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	Type    string            `json:"type,omitempty" yaml:"type,omitempty"` // builtin, stdio (default), http, or sse
	Profile string            `json:"-" yaml:"-"`
	Command string            `json:"command,omitempty" yaml:"command,omitempty"`
	Args    []string          `json:"args,omitempty" yaml:"args,omitempty"`
	URL     string            `json:"url,omitempty" yaml:"url,omitempty"`
	Headers map[string]string `json:"headers,omitempty" yaml:"headers,omitempty"`
}

func (a ACPAgentConfig) IsEnabled() bool {
	return a.Enabled == nil || *a.Enabled
}

// AgentEndpoints returns one built-in agent for every profile followed by the
// explicitly configured external ACP agents.
func AgentEndpoints(cfg *Config) ([]ACPAgentConfig, error) {
	if cfg == nil {
		return nil, nil
	}
	profileIDs := make([]string, 0, len(cfg.Profiles))
	for id := range cfg.Profiles {
		profileIDs = append(profileIDs, id)
	}
	sort.Slice(profileIDs, func(i, j int) bool {
		if profileIDs[i] == "default" {
			return true
		}
		if profileIDs[j] == "default" {
			return false
		}
		return profileIDs[i] < profileIDs[j]
	})

	endpoints := make([]ACPAgentConfig, 0, len(profileIDs)+len(cfg.Agents))
	used := make(map[string]struct{}, cap(endpoints))
	for _, id := range profileIDs {
		if strings.TrimSpace(id) == "" || strings.Contains(id, ":") {
			return nil, fmt.Errorf("invalid profile id %q", id)
		}
		enabled := true
		endpoints = append(endpoints, ACPAgentConfig{
			ID:      id,
			Name:    id,
			Enabled: &enabled,
			Type:    "builtin",
			Profile: id,
			Command: "miya-agent",
			Args:    []string{"acp"},
		})
		used[id] = struct{}{}
	}
	for _, endpoint := range cfg.Agents {
		if endpoint.Type == "builtin" {
			continue
		}
		if _, exists := used[endpoint.ID]; exists {
			return nil, fmt.Errorf("agent id %q conflicts with a profile", endpoint.ID)
		}
		used[endpoint.ID] = struct{}{}
		endpoints = append(endpoints, endpoint)
	}
	return endpoints, nil
}

func (c *Config) UnmarshalJSON(data []byte) error {
	var probe struct {
		Agents json.RawMessage `json:"agents"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	if len(probe.Agents) > 0 {
		var v any
		if err := json.Unmarshal(probe.Agents, &v); err != nil {
			return err
		}
		if _, ok := v.(map[string]any); ok {
			return fmt.Errorf("config field agents must be an array of ACP endpoints; move old agents object to profiles")
		}
	}

	type alias Config
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*c = Config(decoded)
	return nil
}

// ProviderConfig contains API credentials for a providers.
type ProviderConfig struct {
	APIKey   string            `json:"apiKey" yaml:"apiKey"`
	APIBase  string            `json:"apiBase,omitempty" yaml:"apiBase,omitempty"`   // optional custom base URL
	Type     string            `json:"type,omitempty" yaml:"type,omitempty"`         // provider adapter/auth type
	Protocol string            `json:"protocol,omitempty" yaml:"protocol,omitempty"` // optional native wire protocol
	Models   map[string]string `json:"models,omitempty" yaml:"models,omitempty"`     // public name -> upstream model ID
}

// ProfileConfig contains miya-agents runtime defaults.
type ProfileConfig struct {
	Provider            string  `json:"provider" yaml:"provider"`                                           // provider name, e.g. "openai"
	ModelName           string  `json:"model,omitempty" yaml:"model"`                                       // model name, e.g. "deepseek-chat"
	Workspace           string  `json:"workspace,omitempty" yaml:"workspace,omitempty"`                     // defaults to ~/.miya/workspace
	MaxTokens           int     `json:"maxTokens,omitempty" yaml:"maxTokens,omitempty"`                     // defaults to 8192
	Temperature         float64 `json:"temperature,omitempty" yaml:"temperature,omitempty"`                 // defaults to 0.95
	ContextWindowTokens int     `json:"contextWindowTokens,omitempty" yaml:"contextWindowTokens,omitempty"` // defaults to 128000
	ContextWarnRatio    float64 `json:"contextWarnRatio,omitempty" yaml:"contextWarnRatio,omitempty"`       // defaults to 0.9
}

func (ac *ProfileConfig) GetWorkspace() string {
	workspace := expandPath(ac.Workspace)
	if workspace != "" {
		return workspace
	}
	return filepath.Join(ConfigPath, "workspace")
}

// LoggingConfig contains logging configuration.
type LoggingConfig struct {
	Enabled bool   `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	Level   string `json:"level,omitempty" yaml:"level,omitempty"`   // debug, info, warn, error
	Stdout  bool   `json:"stdout,omitempty" yaml:"stdout,omitempty"` // log to stdout
	File    string `json:"file,omitempty" yaml:"file,omitempty"`     // log file path
}

var ConfigPath = DefaultConfigPath()
var ConfigFile = DefaultConfigFile()

func DefaultConfigFile() string {
	return filepath.Join(ConfigPath, "config.json")
}

func DefaultConfigPath() string {
	home, err := os.UserHomeDir()
	if err == nil && home != "" {
		return filepath.Join(home, ".miya")
	}
	if home = os.Getenv("HOME"); home != "" {
		return filepath.Join(home, ".miya")
	}
	if home = os.Getenv("USERPROFILE"); home != "" {
		return filepath.Join(home, ".miya")
	}
	return filepath.Join(".miya")
}

func LoadConfig() (cfg *Config, err error) {
	return LoadConfigFromFile(ConfigFile)
}

func LoadConfigFromFile(path string) (cfg *Config, err error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("config file not found: %s: %w", path, err)
	}
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return NewConfig(), nil
	}
	if err = json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	Normalize(cfg)
	return cfg, nil
}

// expandPath expands ~ to home directory and resolves the path.
func expandPath(path string) string {
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err == nil {
			path = filepath.Join(home, path[1:])
		}
	}
	return path
}

func (c *Config) Save() error {
	return SaveConfigToFile(ConfigFile, c)
}

func SaveConfigToFile(path string, cfg *Config) error {
	if cfg == nil {
		cfg = NewConfig()
	}
	Normalize(cfg)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0600)
}

func NewConfig() *Config {
	cfg := &Config{
		Agents:     []ACPAgentConfig{},
		Profiles:   map[string]*ProfileConfig{},
		Providers:  map[string]*ProviderConfig{},
		McpServers: map[string]*McpServerConfig{},
	}
	Normalize(cfg)
	return cfg
}

func Normalize(cfg *Config) {
	if cfg == nil {
		return
	}
	if cfg.Agents == nil {
		cfg.Agents = []ACPAgentConfig{}
	}
	for i := range cfg.Agents {
		if cfg.Agents[i].Type == "" {
			cfg.Agents[i].Type = "stdio"
		}
	}
	if cfg.Profiles == nil {
		cfg.Profiles = map[string]*ProfileConfig{}
	}
	if cfg.Providers == nil {
		cfg.Providers = map[string]*ProviderConfig{}
	}
	if cfg.McpServers == nil {
		cfg.McpServers = map[string]*McpServerConfig{}
	}
	for id, server := range cfg.McpServers {
		if server.Type == "" {
			if server.URL != "" && server.Command == "" {
				server.Type = "sse"
			} else {
				server.Type = "stdio"
			}
			cfg.McpServers[id] = server
		}
	}
}
