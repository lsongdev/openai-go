package main

import (
	"log"

	"github.com/lsongdev/openai-go/skills"
)

func main() {
	skillsMap, err := skills.LoadSkillsFromDirectory("~/.agents/skills")
	if err != nil {
		log.Fatal(err)
	}
	for _, skill := range skillsMap {
		log.Println(skill.Name, skill.Description)
	}
}
