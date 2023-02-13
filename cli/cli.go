package cli

import (
	"log"

	"github.com/song940/openai-go/openai"
)

func Run() {
	config := openai.Configuration{
		API:    "https://api.lsong.one:8443/openai",
		APIKey: "sk-",
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
		Model:           openai.GPT3_5_Trubo_0301,
		MaxTokens:       2048,
		Temperature:     0,
		NumberOfChoices: 1,
		Messages: []openai.ChatCompletionMessage{
			message,
		},
	}
	res, err := client.CreateChatCompletion(req)
	if err != nil {
		panic(err)
	}
	log.Println(res.Choices[0].Message.Role)
	log.Println(res.Choices[0].Message.Content)
	log.Println(res.Usage.TotalTokens)
}
