package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/lsongdev/miya-agents/acp"
	"github.com/lsongdev/miya-agents/agent"
	"github.com/lsongdev/miya-agents/config"
	"github.com/lsongdev/miya-agents/logging"
	"github.com/lsongdev/miya-agents/mcp"
	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy"
	"github.com/lsongdev/miya-agents/proxy/providers/anthropic/claude"
	"github.com/lsongdev/miya-agents/proxy/providers/codex"
	"github.com/lsongdev/miya-agents/session"
)

// envVars is a custom flag.Value type for parsing multiple environment variables
type envVars []string

func (e *envVars) String() string {
	return strings.Join(*e, ",")
}

func (e *envVars) Set(value string) error {
	*e = append(*e, value)
	return nil
}

func main() {
	if err := logging.SetupFromDefaultConfig("miya-agents"); err != nil {
		log.Printf("[WARN] logging setup failed: %v", err)
	}

	if len(os.Args) < 2 {
		runCommand(nil)
		return
	}

	// Allow top-level flags like `miya -c` or `miya -r <id>` to start REPL.
	if strings.HasPrefix(os.Args[1], "-") && os.Args[1] != "-h" && os.Args[1] != "--help" {
		runCommand(os.Args[1:])
		return
	}

	command := os.Args[1]
	switch command {
	case "run":
		runCommand(os.Args[2:])
	case "sessions":
		sessionsCommand()
	case "serve":
		serveCommand()
	case "mcp":
		mcpCommand()
	case "acp":
		acpCommand()
	case "skills":
		skillsCommand()
	case "profile":
		agentCommand()
	case "models":
		modelsCommand()
	case "provider":
		providerCommand()
	case "onboard":
		onboardCommand()
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Printf("Unknown command: %s\n\n", command)
		printUsage()
	}
}

func acpCommand() {
	acpServeCommand()
}

func acpServeCommand() {
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	agentManager := agent.NewAgentManager(cfg)
	server := acp.NewServer(agentManager)
	if err := server.Serve(); err != nil {
		fmt.Fprintf(os.Stderr, "ACP server error: %v\n", err)
		os.Exit(1)
	}
}

func runCommand(args []string) {
	var (
		agentName    string
		workspace    string
		continueLast bool
		resumeID     string
	)

	flagSet := flag.NewFlagSet("run", flag.ExitOnError)
	flagSet.StringVar(&agentName, "agent", "", "Agent name (default from session or 'default')")
	flagSet.StringVar(&workspace, "workspace", "", "Workspace directory")
	flagSet.BoolVar(&continueLast, "c", false, "Continue the most recent session")
	flagSet.BoolVar(&continueLast, "continue", false, "Continue the most recent session")
	flagSet.StringVar(&resumeID, "r", "", "Resume the session with the given ID")
	flagSet.StringVar(&resumeID, "resume", "", "Resume the session with the given ID")
	flagSet.Parse(args)

	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		fmt.Println("Run 'miya onboard' to set up your first agent.")
		os.Exit(1)
	}

	var sess *session.Session
	switch {
	case resumeID != "":
		s, err := session.Load(resumeID)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
		sess = s
	case continueLast:
		s, err := session.Latest()
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
		if s == nil {
			fmt.Println("No previous session found. Starting a new one.")
		} else {
			sess = s
		}
	}

	if sess != nil && agentName == "" {
		agentName = sess.AgentName
	}
	if agentName == "" {
		agentName = "default"
	}

	agentManager := agent.NewAgentManager(cfg)
	ag, err := agentManager.UseAgent(agentName)
	if err != nil {
		log.Fatalf("profile error: %v", err)
	}

	if sess == nil {
		sess = ag.NewSession()
	}

	output := &stdoutWriter{}
	sink := agent.NewRecordingSink(sess, output)
	ctx := context.Background()
	rl := New("> ")

	fmt.Printf("miya REPL (agent: %s, session: %s) - type 'exit' or 'quit' to leave\n", agentName, sess.ID)
	for {
		line, err := rl.Readline()
		if err != nil {
			if err == ErrInterrupted {
				fmt.Println()
				return
			}
			fmt.Println()
			return
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "exit" || line == "quit" {
			return
		}
		sess.AppendRequest(line)
		agent.RecordUserMessage(sess, line)
		if sess.Title == "" {
			sess.Title = sess.DisplayTitle()
			if sess.Title != "" {
				if err := sink.SessionInfo(agent.SessionInfoEvent{Title: sess.Title}); err != nil {
					fmt.Printf("Error: %v\n", err)
					continue
				}
			}
		}
		if err := sess.Save(); err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		if err := ag.RunAgentLoop(ctx, sess, sink); err != nil {
			fmt.Printf("Error: %v\n", err)
		}
	}
}

type stdoutMode string

const (
	stdoutModeIdle      stdoutMode = ""
	stdoutModeAssistant stdoutMode = "assistant"
	stdoutModeThought   stdoutMode = "thought"
)

type stdoutWriter struct {
	md   *MarkdownRenderer
	mode stdoutMode
}

func (w *stdoutWriter) AssistantDelta(s string) error {
	w.finishThought()
	if w.md == nil {
		w.md = NewMarkdownRenderer(os.Stdout)
	}
	w.mode = stdoutModeAssistant
	w.md.Write(s)
	return nil
}

func (w *stdoutWriter) AssistantFile(event agent.FileEvent) error {
	w.finishThought()
	if w.md != nil {
		w.md.Flush()
	}
	if event.URI != "" {
		fmt.Printf("\n[file] %s (%s, %d bytes): %s\n", event.Name, event.MimeType, event.Size, event.URI)
		return nil
	}
	fmt.Printf("\n[file] %s (%s, %d bytes, inline)\n", event.Name, event.MimeType, event.Size)
	return nil
}

func (w *stdoutWriter) ThoughtDelta(s string) error {
	if s == "" {
		return nil
	}
	if w.md != nil {
		w.md.Flush()
	}
	if w.mode != stdoutModeThought {
		fmt.Printf("\n%s[thought]%s %s", dim, reset, dim)
		w.mode = stdoutModeThought
	}
	fmt.Print(s)
	return nil
}

func (w *stdoutWriter) ToolCallStart(event agent.ToolCallEvent) error {
	w.finishThought()
	if w.md != nil {
		w.md.Flush()
	}
	fmt.Printf("\n[tool] %s(%s)\n", event.Name, event.Arguments)
	return nil
}

func (w *stdoutWriter) ToolCallDone(event agent.ToolCallEvent) error {
	w.finishThought()
	fmt.Printf("[tool:%s] %s\n", event.Status, event.Result)
	return nil
}

func (w *stdoutWriter) SessionInfo(event agent.SessionInfoEvent) error {
	return nil
}

func (w *stdoutWriter) Usage(event agent.UsageEvent) error {
	return nil
}

func (w *stdoutWriter) Done() error {
	w.finishThought()
	if w.md != nil {
		w.md.Flush()
	}
	fmt.Println()
	w.mode = stdoutModeIdle
	return nil
}

func (w *stdoutWriter) finishThought() {
	if w.mode != stdoutModeThought {
		return
	}
	fmt.Printf("%s\n", reset)
	w.mode = stdoutModeIdle
}

func sessionsCommand() {
	if len(os.Args) < 3 {
		sessionsListCommand()
		return
	}

	subcommand := os.Args[2]
	switch subcommand {
	case "list":
		sessionsListCommand()
	default:
		fmt.Printf("Unknown sessions subcommand: %s\n", subcommand)
		fmt.Println("Run 'miya sessions list' to list sessions.")
	}
}

func sessionsListCommand() {
	sessions, err := session.List()
	if err != nil {
		fmt.Printf("Error listing sessions: %v\n", err)
		return
	}
	if len(sessions) == 0 {
		fmt.Println("No sessions found.")
		return
	}

	fmt.Printf("Sessions (%d):\n\n", len(sessions))
	for _, s := range sessions {
		preview := strings.ReplaceAll(s.FirstUserMessage(), "\n", " ")
		if len(preview) > 60 {
			preview = preview[:57] + "..."
		}
		created := "?"
		if !s.CreatedAt.IsZero() {
			created = s.CreatedAt.Format("2006-01-02 15:04")
		}
		fmt.Printf("  %s\n", s.ID)
		fmt.Printf("    agent=%s  messages=%d  created=%s\n", s.AgentName, len(s.Messages), created)
		if preview != "" {
			fmt.Printf("    %s\n", preview)
		}
		fmt.Println()
	}
	fmt.Println("Resume with: miya -r <id>")
}

// addLocalProviders registers providers backed by locally stored OAuth
// credentials (Codex CLI and Claude Code logins) when they exist.
func addLocalProviders(r *proxy.Proxy) int {
	added := 0
	if r.FindProvider("codex") == nil && codex.LoggedIn() {
		if store, err := codex.LoadStore(); err == nil {
			r.AddProvider(codex.Provider(store))
			added++
			log.Printf("[PROXY] registered codex provider (%s)", store.Path())
		} else {
			log.Printf("[PROXY] load codex credentials: %v", err)
		}
	}
	if r.FindProvider("claude") == nil && claude.LoggedIn() {
		if store, err := claude.LoadStore(); err == nil {
			r.AddProvider(claude.Provider(store))
			added++
			log.Printf("[PROXY] registered claude provider (%s)", store.Path())
		} else {
			log.Printf("[PROXY] load claude credentials: %v", err)
		}
	}
	return added
}

func serveCommand() {
	flagSet := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := flagSet.String("addr", ":8090", "Listen address (host:port)")
	flagSet.Parse(os.Args[2:])

	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		os.Exit(1)
	}
	r := proxy.NewProxy()
	providerCount := 0
	for name, p := range cfg.Providers {
		var models []string
		for _, profile := range cfg.Profiles {
			if profile.Provider == name && strings.TrimSpace(profile.ModelName) != "" {
				models = append(models, profile.ModelName)
			}
		}
		for publicModel := range p.Models {
			models = append(models, publicModel)
		}
		provider := &proxy.Provider{
			Name:             name,
			Type:             proxy.ProviderType(p.Type),
			Protocol:         proxy.Protocol(p.Protocol),
			BaseURL:          p.APIBase,
			APIKey:           p.APIKey,
			DefaultMaxTokens: 4096,
			Models:           models,
			ModelAliases:     p.Models,
		}
		r.AddProvider(provider)
		providerCount++
	}
	providerCount += addLocalProviders(r)
	if providerCount == 0 {
		fmt.Println("No providers configured and no local Codex or Claude login found.")
		os.Exit(1)
	}
	r.OnRequest(func(ctx *proxy.RequestContext) error {
		log.Printf("[REQUEST] %s model=%s stream=%v", ctx.RequestID, ctx.Input.Model, ctx.Input.Stream)
		if providerName := strings.TrimSpace(ctx.Request.Header.Get("X-Miya-Provider")); providerName != "" {
			ctx.Upstream = r.FindProvider(providerName)
			if ctx.Upstream == nil {
				return &proxy.RequestError{Status: http.StatusBadRequest, Message: "unknown provider " + providerName}
			}
		} else {
			ctx.Upstream = r.FindProviderForModel(ctx.Input.Model)
		}
		return nil
	})
	r.OnResponse(func(ctx *proxy.ResponseContext) {
		var usage any
		if ctx.Output != nil {
			usage = ctx.Output.Usage
		}
		log.Printf("[RESPONSE] %s usage=%v error=%v", ctx.RequestID, usage, ctx.Error)
	})
	log.Printf("miya serve listening on %s", *addr)
	if err := http.ListenAndServe(*addr, r); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func printUsage() {
	fmt.Println("miya - AI assistant")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  miya [options]            Start REPL (default command)")
	fmt.Println("  miya run [options]        Start REPL")
	fmt.Println("  miya sessions [list]      List previous sessions")
	fmt.Println("  miya serve [--addr ...]   Start the LLM proxy HTTP server")
	fmt.Println("  miya help")
	fmt.Println()
	fmt.Println("REPL Options:")
	fmt.Println("  --agent <name>       Agent name (default: from session or 'default')")
	fmt.Println("  --workspace <path>   Workspace directory")
	fmt.Println("  -c, --continue       Continue the most recent session")
	fmt.Println("  -r, --resume <id>    Resume the session with the given ID")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  miya                              # new session with the default agent")
	fmt.Println("  miya -c                           # continue the most recent session")
	fmt.Println("  miya -r 1f2e3d4c-...              # resume a specific session")
	fmt.Println("  miya run --agent myagent")
	fmt.Println("  miya sessions list")
	fmt.Println("  miya serve --addr :8080")
	fmt.Println("  miya help")
	fmt.Println()
	fmt.Println("MCP Commands:")
	fmt.Println("  miya mcp add <name> --command <cmd> [--args <args>] [--env <KEY=VALUE>]")
	fmt.Println("  miya mcp list")
	fmt.Println("  miya mcp remove <name>")
	fmt.Println()
	fmt.Println("MCP Examples:")
	fmt.Println("  miya mcp add filesystem --command npx --args \"-y @modelcontextprotocol/server-filesystem ~\"")
	fmt.Println("  miya mcp add memory --command npx --args \"-y @modelcontextprotocol/server-memory\"")
	fmt.Println("  miya mcp list")
	fmt.Println("  miya mcp remove filesystem")
	fmt.Println()
	fmt.Println("ACP Commands:")
	fmt.Println("  miya acp          Start ACP server (JSON-RPC over stdio)")
	fmt.Println()
	fmt.Println("Skills Commands:")
	fmt.Println("  miya skills list    List all available skills")
	fmt.Println()
	fmt.Println("Skills Examples:")
	fmt.Println("  miya skills list")
	fmt.Println()
	fmt.Println("Profile Commands:")
	fmt.Println("  miya profile list")
	fmt.Println("  miya profile add <name> --provider <provider> --model <model>")
	fmt.Println()
	fmt.Println("Profile Examples:")
	fmt.Println("  miya profile list")
	fmt.Println("  miya profile add myprofile --provider openai --model gpt-4")
	fmt.Println("  miya profile add coding --provider anthropic --model claude-3-5-sonnet")
	fmt.Println()
	fmt.Println("Models Commands:")
	fmt.Println("  miya models list <provider>    List available models for a provider")
	fmt.Println()
	fmt.Println("Models Examples:")
	fmt.Println("  miya models list openai")
	fmt.Println("  miya models list anthropic")
	fmt.Println()
	fmt.Println("Provider Commands:")
	fmt.Println("  miya provider list")
	fmt.Println("  miya provider add <name> --api-key <key> [--api-base <url>]")
	fmt.Println("  miya provider remove <name>")
	fmt.Println()
	fmt.Println("Provider Examples:")
	fmt.Println("  miya provider list")
	fmt.Println("  miya provider add openai --api-key sk-xxx")
	fmt.Println("  miya provider add anthropic --api-key sk-xxx --api-base https://api.anthropic.com")
	fmt.Println("  miya provider remove openai")
	fmt.Println()
	fmt.Println("Onboard Command:")
	fmt.Println("  miya onboard    Interactive setup for new users")
	fmt.Println()
	fmt.Println("Onboard Example:")
	fmt.Println("  miya onboard")
}

func mcpCommand() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: miya mcp <subcommand> [options]")
		fmt.Println()
		fmt.Println("Subcommands:")
		fmt.Println("  add     Add a new MCP server")
		fmt.Println("  list    List all MCP servers")
		fmt.Println("  remove  Remove an MCP server")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  miya mcp add filesystem --command npx --args \"-y @modelcontextprotocol/server-filesystem ~\"")
		fmt.Println("  miya mcp list")
		fmt.Println("  miya mcp remove filesystem")
		return
	}

	subcommand := os.Args[2]
	switch subcommand {
	case "add":
		mcpAddCommand(os.Args[3:])
	case "list":
		mcpListCommand()
	case "remove":
		mcpRemoveCommand(os.Args[3:])
	default:
		fmt.Printf("Unknown mcp subcommand: %s\n", subcommand)
		fmt.Println("Run 'miya mcp' for usage.")
	}
}

func mcpAddCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: miya mcp add <name> [--type stdio|http|sse] [--command <cmd>] [--url <url>] [--args <args>] [--env <KEY=VALUE>...] [--header <KEY=VALUE>...]")
		fmt.Println()
		fmt.Println("Options:")
		fmt.Println("  --type <type>     Transport type: stdio, http, or sse (default: stdio)")
		fmt.Println("  --command <cmd>   Command for stdio MCP servers")
		fmt.Println("  --url <url>       URL for HTTP/SSE MCP servers")
		fmt.Println("  --args <args>     Arguments for the command (space-separated)")
		fmt.Println("  --env <KEY=VALUE> Environment variables (can be specified multiple times)")
		fmt.Println("  --header <KEY=VALUE> HTTP headers for HTTP/SSE servers")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  miya mcp add filesystem --command npx --args \"-y @modelcontextprotocol/server-filesystem ~\"")
		fmt.Println("  miya mcp add remote --type sse --url https://example.com/sse --header Authorization=Bearer_xxx")
		fmt.Println("  miya mcp add myserver --command python --args \"server.py\" --env API_KEY=abc123 --env DEBUG=true")
		return
	}

	name := args[0]
	if name == "" {
		fmt.Println("Error: server name cannot be empty")
		return
	}

	flagSet := flag.NewFlagSet("mcp add", flag.ExitOnError)
	transportType := flagSet.String("type", "stdio", "Transport type: stdio, http, or sse")
	command := flagSet.String("command", "", "Command to run the MCP server")
	serverURL := flagSet.String("url", "", "URL for HTTP/SSE MCP servers")
	argsStr := flagSet.String("args", "", "Arguments for the command")

	var envs envVars
	flagSet.Var(&envs, "env", "Environment variables (KEY=VALUE)")
	var headers envVars
	flagSet.Var(&headers, "header", "HTTP headers (KEY=VALUE)")

	flagSet.Parse(args[1:])

	kind := strings.ToLower(strings.TrimSpace(*transportType))
	if kind == "" {
		kind = "stdio"
	}
	if kind != "stdio" && kind != "http" && kind != "sse" {
		fmt.Println("Error: --type must be one of stdio, http, or sse")
		return
	}
	if kind == "stdio" && *command == "" {
		fmt.Println("Error: --command is required for stdio MCP servers")
		return
	}
	if (kind == "http" || kind == "sse") && *serverURL == "" {
		fmt.Println("Error: --url is required for HTTP/SSE MCP servers")
		return
	}

	// Parse args string into slice
	var cmdArgs []string
	if *argsStr != "" {
		cmdArgs = strings.Split(*argsStr, " ")
	}

	// Parse env vars
	envMap := make(map[string]string)
	for _, env := range envs {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) != 2 {
			fmt.Printf("Warning: invalid env format '%s', expected KEY=VALUE\n", env)
			continue
		}
		envMap[parts[0]] = parts[1]
	}
	headerMap := make(map[string]string)
	for _, header := range headers {
		parts := strings.SplitN(header, "=", 2)
		if len(parts) != 2 {
			fmt.Printf("Warning: invalid header format '%s', expected KEY=VALUE\n", header)
			continue
		}
		headerMap[parts[0]] = parts[1]
	}

	// Load existing config
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	// Check if server already exists
	if cfg.McpServers == nil {
		cfg.McpServers = make(map[string]*mcp.McpServerConfig)
	}
	if _, exists := cfg.McpServers[name]; exists {
		fmt.Printf("Warning: MCP server '%s' already exists, updating...\n", name)
	}

	// Add/update the MCP server
	cfg.McpServers[name] = &mcp.McpServerConfig{
		Type:    kind,
		Command: *command,
		Args:    cmdArgs,
		Env:     envMap,
		URL:     *serverURL,
		Headers: headerMap,
	}

	// Save config
	if err := saveConfig(cfg); err != nil {
		fmt.Printf("Error saving config: %v\n", err)
		return
	}

	fmt.Printf("Successfully added MCP server '%s'\n", name)
	fmt.Printf("  Type: %s\n", kind)
	if *command != "" {
		fmt.Printf("  Command: %s\n", *command)
	}
	if *serverURL != "" {
		fmt.Printf("  URL: %s\n", *serverURL)
	}
	if len(cmdArgs) > 0 {
		fmt.Printf("  Args: %s\n", *argsStr)
	}
	if len(envMap) > 0 {
		fmt.Printf("  Env: %v\n", envMap)
	}
	if len(headerMap) > 0 {
		fmt.Printf("  Headers: %v\n", headerMap)
	}
}

func mcpListCommand() {
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	if len(cfg.McpServers) == 0 {
		fmt.Println("No MCP servers configured.")
		return
	}

	fmt.Println("MCP Servers:")
	fmt.Println()
	for name, server := range cfg.McpServers {
		fmt.Printf("  %s:\n", name)
		kind := server.Type
		if kind == "" {
			kind = "stdio"
		}
		fmt.Printf("    Type: %s\n", kind)
		if server.Command != "" {
			fmt.Printf("    Command: %s\n", server.Command)
		}
		if server.URL != "" {
			fmt.Printf("    URL: %s\n", server.URL)
		}
		if len(server.Args) > 0 {
			fmt.Printf("    Args: %s\n", strings.Join(server.Args, " "))
		}
		if len(server.Env) > 0 {
			fmt.Printf("    Env: %v\n", server.Env)
		}
		if len(server.Headers) > 0 {
			fmt.Printf("    Headers: %v\n", server.Headers)
		}
		fmt.Println()
	}
}

func mcpRemoveCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: miya mcp remove <name>")
		fmt.Println()
		fmt.Println("Example:")
		fmt.Println("  miya mcp remove filesystem")
		return
	}

	name := args[0]

	// Load existing config
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	if len(cfg.McpServers) == 0 {
		fmt.Println("No MCP servers configured.")
		return
	}

	if _, exists := cfg.McpServers[name]; !exists {
		fmt.Printf("MCP server '%s' not found.\n", name)
		return
	}

	delete(cfg.McpServers, name)

	// Save config
	if err := saveConfig(cfg); err != nil {
		fmt.Printf("Error saving config: %v\n", err)
		return
	}

	fmt.Printf("Successfully removed MCP server '%s'\n", name)
}

func saveConfig(cfg *config.Config) error {
	return config.SaveConfigToFile(config.ConfigFile, cfg)
}

func skillsCommand() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: miya skills <subcommand> [options]")
		fmt.Println()
		fmt.Println("Subcommands:")
		fmt.Println("  list    List all available skills")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  miya skills list")
		return
	}

	subcommand := os.Args[2]
	switch subcommand {
	case "list":
		skillsListCommand()
	default:
		fmt.Printf("Unknown skills subcommand: %s\n", subcommand)
		fmt.Println("Run 'miya skills' for usage.")
	}
}

func skillsListCommand() {
	skillsDir := filepath.Join(config.ConfigPath, "skills")

	// Check if skills directory exists
	if _, err := os.Stat(skillsDir); os.IsNotExist(err) {
		fmt.Println("No skills directory found.")
		return
	}

	// Read skills directory
	entries, err := os.ReadDir(skillsDir)
	if err != nil {
		fmt.Printf("Error reading skills directory: %v\n", err)
		return
	}

	if len(entries) == 0 {
		fmt.Println("No skills found.")
		return
	}

	fmt.Println("Available Skills:")
	fmt.Println()

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		skillName := entry.Name()
		skillPath := filepath.Join(skillsDir, skillName)
		skillFile := filepath.Join(skillPath, "SKILL.md")

		// Read skill description from SKILL.md
		description := ""
		if data, err := os.ReadFile(skillFile); err == nil {
			content := string(data)
			// Extract description from frontmatter
			if idx := strings.Index(content, "description:"); idx != -1 {
				start := idx + len("description:")
				end := strings.Index(content[start:], "\n")
				if end != -1 {
					description = strings.TrimSpace(content[start : start+end])
				}
			}
		}

		fmt.Printf("  %s\n", skillName)
		if description != "" {
			// Wrap long descriptions
			if len(description) > 80 {
				description = description[:77] + "..."
			}
			fmt.Printf("    %s\n", description)
		}
		fmt.Println()
	}
}

func agentCommand() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: miya profile <subcommand> [options]")
		fmt.Println()
		fmt.Println("Subcommands:")
		fmt.Println("  list    List all configured profiles")
		fmt.Println("  add     Add a new profile")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  miya profile list")
		fmt.Println("  miya profile add myprofile --provider openai --model gpt-4")
		return
	}

	subcommand := os.Args[2]
	switch subcommand {
	case "list":
		agentListCommand()
	case "add":
		agentAddCommand(os.Args[3:])
	default:
		fmt.Printf("Unknown agent subcommand: %s\n", subcommand)
		fmt.Println("Run 'miya profile' for usage.")
	}
}

func agentListCommand() {
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	if len(cfg.Profiles) == 0 {
		fmt.Println("No profiles configured.")
		return
	}

	fmt.Println("Configured Profiles:")
	fmt.Println()
	for name, agent := range cfg.Profiles {
		fmt.Printf("  %s:\n", name)
		fmt.Printf("    Provider: %s\n", agent.Provider)
		fmt.Printf("    Model: %s\n", agent.ModelName)
		if agent.Workspace != "" {
			fmt.Printf("    Workspace: %s\n", agent.Workspace)
		}
		fmt.Println()
	}
}

func agentAddCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: miya profile add <name> --provider <provider> --model <model>")
		fmt.Println()
		fmt.Println("Options:")
		fmt.Println("  --provider <provider>   The provider name (required)")
		fmt.Println("  --model <model>         The model name (required)")
		fmt.Println("  --workspace <path>      Workspace directory (optional)")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  miya profile add myprofile --provider openai --model gpt-4")
		fmt.Println("  miya profile add coding --provider anthropic --model claude-3-5-sonnet")
		fmt.Println("  miya profile add dev --provider openai --model gpt-4 --workspace ~/projects/mybot")
		return
	}

	name := args[0]
	if name == "" {
		fmt.Println("Error: profile name cannot be empty")
		return
	}

	flagSet := flag.NewFlagSet("profile add", flag.ExitOnError)
	provider := flagSet.String("provider", "", "Provider name (required)")
	model := flagSet.String("model", "", "Model name (required)")
	workspace := flagSet.String("workspace", "", "Workspace directory")

	flagSet.Parse(args[1:])

	if *provider == "" {
		fmt.Println("Error: --provider is required")
		return
	}

	if *model == "" {
		fmt.Println("Error: --model is required")
		return
	}

	// Load existing config
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	// Check if profile already exists
	if cfg.Profiles == nil {
		cfg.Profiles = make(map[string]*config.ProfileConfig)
	}
	if _, exists := cfg.Profiles[name]; exists {
		fmt.Printf("Warning: profile '%s' already exists, updating...\n", name)
	}

	// Add/update the profile
	cfg.Profiles[name] = &config.ProfileConfig{
		Provider:  *provider,
		ModelName: *model,
		Workspace: *workspace,
	}

	// Save config
	if err := saveConfig(cfg); err != nil {
		fmt.Printf("Error saving config: %v\n", err)
		return
	}

	fmt.Printf("Successfully added profile '%s'\n", name)
	fmt.Printf("  Provider: %s\n", *provider)
	fmt.Printf("  Model: %s\n", *model)
	if *workspace != "" {
		fmt.Printf("  Workspace: %s\n", *workspace)
	}
}

func modelsCommand() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: miya models <subcommand> [options]")
		fmt.Println()
		fmt.Println("Subcommands:")
		fmt.Println("  list <provider>    List available models for a provider")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  miya models list openai")
		fmt.Println("  miya models list anthropic")
		return
	}

	subcommand := os.Args[2]
	switch subcommand {
	case "list":
		modelsListCommand(os.Args[3:])
	default:
		fmt.Printf("Unknown models subcommand: %s\n", subcommand)
		fmt.Println("Run 'miya models' for usage.")
	}
}

func modelsListCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: miya models list <provider>")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  miya models list openai")
		fmt.Println("  miya models list anthropic")
		return
	}

	providerName := args[0]

	// Load config to get provider
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	provider, exists := cfg.Providers[providerName]
	if !exists {
		fmt.Printf("Provider '%s' not found in config.\n", providerName)
		fmt.Println("Use 'miya provider list' to see configured providers.")
		return
	}

	// Create OpenAI client and fetch models
	client, err := openai.NewClient(&openai.Configuration{
		APIKey: provider.APIKey,
		API:    provider.APIBase,
	})
	if err != nil {
		fmt.Printf("Error creating client: %v\n", err)
		return
	}

	models, err := client.Models()
	if err != nil {
		fmt.Printf("Error fetching models: %v\n", err)
		return
	}

	fmt.Printf("Available models for provider '%s':\n", providerName)
	fmt.Println()
	for _, model := range models {
		fmt.Printf("  %s\n", model.ID)
	}
	fmt.Println()
}

func providerCommand() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: miya provider <subcommand> [options]")
		fmt.Println()
		fmt.Println("Subcommands:")
		fmt.Println("  list    List all configured providers")
		fmt.Println("  add     Add a new provider")
		fmt.Println("  remove  Remove a provider")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  miya provider list")
		fmt.Println("  miya provider add openai --api-key sk-xxx")
		fmt.Println("  miya provider remove openai")
		return
	}

	subcommand := os.Args[2]
	switch subcommand {
	case "list":
		providerListCommand()
	case "add":
		providerAddCommand(os.Args[3:])
	case "remove":
		providerRemoveCommand(os.Args[3:])
	default:
		fmt.Printf("Unknown provider subcommand: %s\n", subcommand)
		fmt.Println("Run 'miya provider' for usage.")
	}
}

func providerListCommand() {
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	if len(cfg.Providers) == 0 {
		fmt.Println("No providers configured.")
		return
	}

	fmt.Println("Configured Providers:")
	fmt.Println()
	for name, provider := range cfg.Providers {
		fmt.Printf("  %s:\n", name)
		fmt.Printf("    API Key: %s\n", maskAPIKey(provider.APIKey))
		if provider.APIBase != "" {
			fmt.Printf("    API Base: %s\n", provider.APIBase)
		}
		if provider.Type != "" {
			fmt.Printf("    Type: %s\n", provider.Type)
		}
		fmt.Println()
	}
}

func maskAPIKey(key string) string {
	if len(key) <= 8 {
		return "***"
	}
	return key[:4] + "..." + key[len(key)-4:]
}

func providerAddCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: miya provider add <name> --api-key <key> [--api-base <url>] [--type <openai|anthropic>]")
		fmt.Println()
		fmt.Println("Options:")
		fmt.Println("  --api-key <key>       The API key (required)")
		fmt.Println("  --api-base <url>      The API base URL (optional)")
		fmt.Println("  --type <type>         Provider protocol: openai (default) or anthropic")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  miya provider add openai --api-key sk-xxx")
		fmt.Println("  miya provider add anthropic --api-key sk-xxx --type anthropic")
		return
	}

	name := args[0]
	if name == "" {
		fmt.Println("Error: provider name cannot be empty")
		return
	}

	flagSet := flag.NewFlagSet("provider add", flag.ExitOnError)
	apiKey := flagSet.String("api-key", "", "API key (required)")
	apiBase := flagSet.String("api-base", "", "API base URL")
	providerType := flagSet.String("type", "", "Provider protocol: openai or anthropic")

	flagSet.Parse(args[1:])

	if *apiKey == "" {
		fmt.Println("Error: --api-key is required")
		return
	}

	// Load existing config
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	// Check if provider already exists
	if cfg.Providers == nil {
		cfg.Providers = make(map[string]*config.ProviderConfig)
	}
	if _, exists := cfg.Providers[name]; exists {
		fmt.Printf("Warning: provider '%s' already exists, updating...\n", name)
	}

	// Add/update the provider
	cfg.Providers[name] = &config.ProviderConfig{
		APIKey:  *apiKey,
		APIBase: *apiBase,
		Type:    *providerType,
	}

	// Save config
	if err := saveConfig(cfg); err != nil {
		fmt.Printf("Error saving config: %v\n", err)
		return
	}

	fmt.Printf("Successfully added provider '%s'\n", name)
	fmt.Printf("  API Key: %s\n", maskAPIKey(*apiKey))
	if *apiBase != "" {
		fmt.Printf("  API Base: %s\n", *apiBase)
	}
	if *providerType != "" {
		fmt.Printf("  Type: %s\n", *providerType)
	}
}

func providerRemoveCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: miya provider remove <name>")
		fmt.Println()
		fmt.Println("Example:")
		fmt.Println("  miya provider remove openai")
		return
	}

	name := args[0]

	// Load existing config
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	if len(cfg.Providers) == 0 {
		fmt.Println("No providers configured.")
		return
	}

	if _, exists := cfg.Providers[name]; !exists {
		fmt.Printf("Provider '%s' not found.\n", name)
		return
	}

	delete(cfg.Providers, name)

	// Save config
	if err := saveConfig(cfg); err != nil {
		fmt.Printf("Error saving config: %v\n", err)
		return
	}

	fmt.Printf("Successfully removed provider '%s'\n", name)
}

func onboardCommand() {
	fmt.Println("Welcome to miya!")
	fmt.Println()
	fmt.Println("This interactive setup will help you configure your first agent.")
	fmt.Println()

	// Step 1: Setup provider
	fmt.Println("Step 1: Configure a Provider")
	fmt.Println("-----------------------------")
	fmt.Println("A provider is an AI service that powers your agent (e.g., OpenAI, Anthropic, DeepSeek).")
	fmt.Println()

	providerName := promptInput("Enter provider name (e.g., openai, anthropic, deepseek): ")
	if providerName == "" {
		fmt.Println("Provider name is required. Exiting setup.")
		return
	}

	apiKey := promptInput("Enter your API key: ")
	if apiKey == "" {
		fmt.Println("API key is required. Exiting setup.")
		return
	}

	apiBase := promptInput("Enter API base URL (optional, press Enter to skip): ")

	// Step 2: Setup agent
	fmt.Println()
	fmt.Println("Step 2: Configure Your First Agent")
	fmt.Println("-----------------------------------")
	fmt.Println("An agent is a configuration that connects a provider to a specific model.")
	fmt.Println()

	agentName := promptInput("Enter profile name (e.g., default, assistant): ")
	if agentName == "" {
		agentName = "default"
	}

	modelName := promptInput("Enter model name (e.g., gpt-4, claude-3-5-sonnet, deepseek-chat): ")
	if modelName == "" {
		fmt.Println("Model name is required. Exiting setup.")
		return
	}

	workspace := promptInput("Enter workspace directory (optional, press Enter to skip): ")

	// Create config
	fmt.Println()
	fmt.Println("Setting up your configuration...")

	// Ensure config directory exists
	if err := os.MkdirAll(config.ConfigPath, 0755); err != nil {
		fmt.Printf("Error creating config directory: %v\n", err)
		return
	}

	// Load existing config or create new one
	cfg, err := config.LoadConfig()
	if err != nil {
		// Create new config if none exists
		cfg = &config.Config{
			Profiles:  make(map[string]*config.ProfileConfig),
			Providers: make(map[string]*config.ProviderConfig),
		}
	}

	// Add provider
	if cfg.Providers == nil {
		cfg.Providers = make(map[string]*config.ProviderConfig)
	}
	cfg.Providers[providerName] = &config.ProviderConfig{
		APIKey:  apiKey,
		APIBase: apiBase,
	}

	// Add profile
	if cfg.Profiles == nil {
		cfg.Profiles = make(map[string]*config.ProfileConfig)
	}
	cfg.Profiles[agentName] = &config.ProfileConfig{
		Provider:  providerName,
		ModelName: modelName,
		Workspace: workspace,
	}

	// Save config
	if err := saveConfig(cfg); err != nil {
		fmt.Printf("Error saving config: %v\n", err)
		return
	}

	fmt.Println()
	fmt.Println("Setup complete!")
	fmt.Println()
	fmt.Println("Configuration Summary:")
	fmt.Printf("  Provider: %s\n", providerName)
	if apiBase != "" {
		fmt.Printf("  API Base: %s\n", apiBase)
	}
	fmt.Printf("  Profile: %s\n", agentName)
	fmt.Printf("  Model: %s\n", modelName)
	if workspace != "" {
		fmt.Printf("  Workspace: %s\n", workspace)
	}
	fmt.Println()
	fmt.Println("You can now run miya with:")
	fmt.Printf("  miya run --agent %s\n", agentName)
	fmt.Println()
	fmt.Println("Or run in terminal mode:")
	fmt.Printf("  miya run --agent %s --terminal\n", agentName)
	fmt.Println()
}

func promptInput(prompt string) string {
	fmt.Print(prompt)
	var input string
	fmt.Scanln(&input)
	return strings.TrimSpace(input)
}
