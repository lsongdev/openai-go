package cli

import (
	"log"

	"github.com/song940/openai-go/openai"
)

func Run() {
	config := openai.Configuration{
		API:    "https://oai.lsong.org/v1",
		APIKey: "5e88a1c714c84cbba4e29e4a956d0f7b",
	}
	client, err := openai.NewClient(config)
	if err != nil {
		log.Fatal(err)
	}
	message := openai.ChatCompletionMessage{
		Role:    openai.RoleUser,
		Content: "Hello!",
	}
	req := openai.ChatCompletionRequest{
		Model:           openai.GPT3_5_Trubo,
		MaxTokens:       2048,
		Temperature:     0,
		NumberOfChoices: 1,
		Messages: []openai.ChatCompletionMessage{
			message,
		},
	}
	res, err := client.CreateChatCompletion(req)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("[%s](%d): %s\n", res.Choices[0].Message.Role, res.Usage.TotalTokens, res.Choices[0].Message.Content)
}
