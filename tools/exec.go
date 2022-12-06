package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/lsongdev/openai-go/openai"
)

const (
	ExecDefaultTimeoutSeconds = 60
	execOutputMaxChars        = 50000
)

// ExecTool executes shell commands.
type ExecTool struct {
	Workspace           string
	DefaultTimeout      int
	RestrictToWorkspace bool
}

// Def returns the tool definition.
func (t *ExecTool) Def() openai.ToolDef {
	return openai.ToolDef{
		Type: "function",
		Function: openai.FunctionDef{
			Name:        "exec",
			Description: "Execute a shell command and return its output. Use for running programs, scripts, git commands, etc.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "The shell command to execute.",
					},
					"workdir": map[string]any{
						"type":        "string",
						"description": "Optional working directory. Defaults to workspace.",
					},
					"timeout": map[string]any{
						"type":        "integer",
						"description": "Optional timeout in seconds. Defaults to 60.",
					},
				},
				"required": []string{"command"},
			},
		},
	}
}

// execArgs are the arguments for exec.
type execArgs struct {
	Command string `json:"command"`
	Workdir string `json:"workdir,omitempty"`
	Timeout int    `json:"timeout,omitempty"`
}

// Run executes the tool.
func (t *ExecTool) Run(ctx context.Context, args string) string {
	var a execArgs
	if err := json.Unmarshal([]byte(args), &a); err != nil {
		return fmt.Sprintf("Error: failed to parse arguments: %v", err)
	}

	timeout := a.Timeout
	if timeout <= 0 {
		if t.DefaultTimeout > 0 {
			timeout = t.DefaultTimeout
		} else {
			timeout = ExecDefaultTimeoutSeconds
		}
	}

	execCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(execCtx, "sh", "-c", a.Command)
	if a.Workdir != "" {
		cmd.Dir = expandPath(a.Workdir)
	} else if t.Workspace != "" {
		cmd.Dir = t.Workspace
	}

	if t.RestrictToWorkspace && t.Workspace != "" {
		if !isPathWithinWorkspace(cmd.Dir, t.Workspace) {
			return fmt.Sprintf("Error: working directory %q is outside workspace %q (restrictToWorkspace is enabled)", cmd.Dir, t.Workspace)
		}
	}

	output, err := cmd.CombinedOutput()
	if execCtx.Err() == context.DeadlineExceeded {
		return fmt.Sprintf("Error: command timed out after %d seconds\nPartial output:\n%s", timeout, string(output))
	}

	if err != nil {
		return fmt.Sprintf("Command failed: %v\nOutput:\n%s", err, string(output))
	}

	result := string(output)
	if result == "" {
		return "(no output)"
	}
	if len(result) > execOutputMaxChars {
		result = truncateWithNotice(result, execOutputMaxChars)
	}

	return result
}

// expandPath expands a path that may start with ~ to the user's home directory.
func expandPath(path string) string {
	if path == "" {
		return ""
	}
	if path == "~" {
		home, _ := os.UserHomeDir()
		return home
	}
	if strings.HasPrefix(path, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, path[2:])
	}
	return path
}

// isPathWithinWorkspace checks if a path is within the workspace directory.
func isPathWithinWorkspace(path, workspace string) bool {
	if path == "" {
		path, _ = os.Getwd()
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	absPath, err = filepath.EvalSymlinks(absPath)
	if err != nil {
		return false
	}

	absWorkspace, err := filepath.Abs(workspace)
	if err != nil {
		return false
	}
	absWorkspace, err = filepath.EvalSymlinks(absWorkspace)
	if err != nil {
		return false
	}

	if absPath == absWorkspace {
		return true
	}
	return strings.HasPrefix(absPath+string(filepath.Separator), absWorkspace+string(filepath.Separator))
}

// truncateWithNotice truncates content to maxChars and appends a notice.
func truncateWithNotice(content string, maxChars int) string {
	if len(content) <= maxChars {
		return content
	}
	return content[:maxChars] + "\n\n[Content truncated due to size limit.]"
}
