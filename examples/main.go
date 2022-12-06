package main

import (
	"log"

	"github.com/lsongdev/openai-go/openai"
)

func main() {
	client, err := openai.NewClient(&openai.Configuration{
		API:    "https://api.deepseek.com",
		APIKey: "sk-xxx",
	})
	if err != nil {
		log.Fatal(err)
	}

	models, err := client.Models()
	if err != nil {
		log.Fatal(err)
	}
	for _, model := range models {
		log.Printf("Model: %s\n", model.ID)
	}
	message := openai.ChatCompletionMessage{
		Role: openai.RoleUser,
		// Content: "what is the current time?",
		Content: "Hello!",
	}
	tools := []openai.ToolDef{
		{
			Type: "function",
			Function: openai.FunctionDef{
				Name:        "get_current_time",
				Description: "Get the current time",
			},
		},
	}
	request := &openai.ChatCompletionRequest{
		Model: "deepseek-reasoner",
		// MaxTokens: 2048,
		// Temperature:     0,
		// NumberOfChoices: 1,
		Messages: []openai.ChatCompletionMessage{
			message,
		},
		Tools:  tools,
		Stream: true,
	}

	if request.Stream {
		resp, err := client.CreateChatCompletionStream(request)
		if err != nil {
			log.Fatal(err)
		}
		for r := range resp {
			printResponse(&r)
		}
	} else {
		resp, err := client.CreateChatCompletion(request)
		if err != nil {
			log.Fatal(err)
		}
		printResponse(&resp)
	}

}

func printResponse(resp *openai.ChatCompletionResponse) {
	message := resp.GetMessage()
	if message.ReasoningContent != "" {
		log.Printf("Reasoning: %s\n", resp.GetMessage().ReasoningContent)
	}
	log.Println("Content:", message.Content)
	choice := resp.GetFirstChoice()
	if choice.FinishReason != "" {
		log.Println("FinishReason:", choice.FinishReason)
		log.Println("Usage", resp.Usage)
	}
}
