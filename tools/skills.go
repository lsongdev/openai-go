package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lsongdev/openai-go/openai"
	"github.com/lsongdev/openai-go/skills"
)

// SkillsTool provides access to skills.
type SkillsTool struct {
	skills    map[string]*skills.Skill
	Workspace string
}

func (t *SkillsTool) reloadSkillsFromDirectory() (err error) {
	t.skills = map[string]*skills.Skill{}
	files, err := skills.LoadSkillsFromDirectory(t.Workspace)
	if err != nil {
		return
	}
	for _, skill := range files {
		t.skills[skill.Name] = skill
	}
	return
}

// Def returns the tool definition.
func (t *SkillsTool) Def() openai.ToolDef {
	return openai.ToolDef{
		Type: "function",
		Function: openai.FunctionDef{
			Name:        "use_skills",
			Description: "List all available skills or get the full prompt content of a specific skill. Call without 'name' to list all skills, or with 'name' to get a specific skill's content.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type":        "string",
						"description": "Optional. The name of the skill to use. If omitted or empty, lists all available skills.",
					},
				},
			},
		},
	}
}

// useSkillsArgs are the arguments for use_skills.
type useSkillsArgs struct {
	Name string `json:"name,omitempty"`
}

// Run executes the tool.
func (t *SkillsTool) Run(ctx context.Context, args string) string {
	if len(t.skills) == 0 {
		return "No skills registered."
	}
	var a useSkillsArgs
	if err := json.Unmarshal([]byte(args), &a); err != nil {
		return fmt.Sprintf("Error: failed to parse arguments: %v", err)
	}
	// If no name provided, list all skills
	if a.Name == "" {
		err := t.reloadSkillsFromDirectory()
		if err != nil {
			return fmt.Sprintf("Error: failed to load skills: %v", err)
		}
		var sb strings.Builder
		sb.WriteString("Available skills:\n\n")
		for _, s := range t.skills {
			sb.WriteString(fmt.Sprintf("- **%s**: %s\n", s.Name, s.Description))
		}
		return sb.String()
	}

	// Get specific skill
	skill, ok := t.skills[a.Name]
	if !ok {
		availableNames := make([]string, 0, len(t.skills))
		for name := range t.skills {
			availableNames = append(availableNames, name)
		}
		return fmt.Sprintf("Error: skill %q not found. Available skills: %s", a.Name, strings.Join(availableNames, ", "))
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# Skill: %s\n\n", skill.Name))
	if skill.Description != "" {
		sb.WriteString(fmt.Sprintf("**Description**: %s\n\n", skill.Description))
	}
	if len(skill.Metadata) > 0 {
		sb.WriteString("**Metadata**:\n")
		for k, v := range skill.Metadata {
			sb.WriteString(fmt.Sprintf("- %s: %v\n", k, v))
		}
		sb.WriteString("\n")
	}
	sb.WriteString("**Prompt**:\n\n")
	sb.WriteString(skill.Prompt)

	return sb.String()
}
