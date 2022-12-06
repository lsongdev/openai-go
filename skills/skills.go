// Package skills provides the skill system for nagobot.
// Skills are reusable prompt templates that can be loaded dynamically.
package skills

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// Skill represents a skill definition.
// https://agentskills.io/specification
type Skill struct {
	Name        string         `yaml:"name"`
	Description string         `yaml:"description"`
	Metadata    map[string]any `yaml:"metadata,omitempty"`
	Prompt      string         `yaml:"prompt"`
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

func LoadSkillsFromDirectory(dir string) (skills []*Skill, err error) {
	dir = expandPath(dir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			// Try to load as directory-based skill (e.g., /my-skill/SKILL.md)
			skill, loadErr := loadDirectorySkill(dir, entry.Name())
			if loadErr != nil {
				return nil, fmt.Errorf("failed to load skill from directory %s: %w", entry.Name(), loadErr)
			}
			if skill != nil {
				skills = append(skills, skill)
			}
			continue
		}

		name := entry.Name()
		ext := strings.ToLower(filepath.Ext(name))

		var skill *Skill
		var loadErr error

		switch ext {
		case ".yaml", ".yml":
			skill, loadErr = loadYAMLSkill(filepath.Join(dir, name))
		case ".md":
			skill, loadErr = loadMarkdownSkill(filepath.Join(dir, name))
		default:
			continue
		}

		if loadErr != nil {
			return nil, fmt.Errorf("failed to load skill %s: %w", name, loadErr)
		}
		if skill != nil {
			skills = append(skills, skill)
		}
	}
	return
}

// loadDirectorySkill loads a skill from a directory containing SKILL.md.
// Expected format: /path/to/my-skill/SKILL.md
// The directory name is used as the skill name if not specified in frontmatter.
func loadDirectorySkill(dir, skillDirName string) (*Skill, error) {
	skillPath := filepath.Join(dir, skillDirName, "SKILL.md")

	data, err := os.ReadFile(skillPath)
	if err != nil {
		if os.IsNotExist(err) {
			// No SKILL.md in directory, skip it
			return nil, nil
		}
		return nil, err
	}

	content := string(data)
	skill, contentBody, err := ParseSkillFrontmatter(content)
	if err != nil {
		return nil, err
	}

	// No frontmatter, create skill with entire content as prompt
	if skill == nil {
		skill = &Skill{
			Prompt: content,
		}
	} else {
		skill.Prompt = contentBody
	}

	// Default name from directory name
	if skill.Name == "" {
		skill.Name = skillDirName
	}
	return skill, nil
}

// loadYAMLSkill loads a skill from a YAML file.
func loadYAMLSkill(path string) (*Skill, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var skill Skill
	if err := yaml.Unmarshal(data, &skill); err != nil {
		return nil, err
	}

	if skill.Name == "" {
		// Use filename as name
		skill.Name = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}

	return &skill, nil
}

// loadMarkdownSkill loads a skill from a Markdown file with YAML frontmatter.
// Format:
// ---
// name: skill-name
// description: Short description
// tags: [tag1, tag2]
// ---
// # Skill Prompt Content
// The rest of the markdown is the prompt.
func loadMarkdownSkill(path string) (*Skill, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	content := string(data)
	skill, contentBody, err := ParseSkillFrontmatter(content)
	if err != nil {
		return nil, err
	}

	// No frontmatter, treat entire file as prompt
	if skill == nil {
		name := strings.TrimSuffix(filepath.Base(path), ".md")
		return &Skill{
			Name:   name,
			Prompt: content,
		}, nil
	}

	skill.Prompt = contentBody

	// Default name from filename
	if skill.Name == "" {
		skill.Name = strings.TrimSuffix(filepath.Base(path), ".md")
	}

	return skill, nil
}

// FrontmatterResult holds the result of parsing frontmatter.
type FrontmatterResult struct {
	Header  string
	Content string
}

// ParseFrontmatter parses YAML frontmatter from content.
// Returns the header and the remaining content.
// If no frontmatter is found, returns empty header and original content.
//
// Format:
// ---
// name: skill-name
// description: Short description
// tags: [tag1, tag2]
// ---
// # Content
func ParseFrontmatter(content string) (*FrontmatterResult, error) {
	if !strings.HasPrefix(content, "---") {
		return &FrontmatterResult{
			Header:  "",
			Content: content,
		}, nil
	}

	// Split on the closing ---
	parts := strings.SplitN(content[3:], "---", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid frontmatter format: missing closing ---")
	}

	return &FrontmatterResult{
		Header:  strings.TrimSpace(parts[0]),
		Content: strings.TrimSpace(parts[1]),
	}, nil
}

// ParseSkillFrontmatter parses YAML frontmatter into a Skill struct.
// Returns the parsed skill and the remaining content.
func ParseSkillFrontmatter(content string) (*Skill, string, error) {
	result, err := ParseFrontmatter(content)
	if err != nil {
		return nil, "", err
	}

	if result.Header == "" {
		// No frontmatter found
		return nil, result.Content, nil
	}

	var skill Skill
	if err := yaml.Unmarshal([]byte(result.Header), &skill); err != nil {
		return nil, "", fmt.Errorf("invalid YAML frontmatter: %w", err)
	}

	return &skill, result.Content, nil
}
