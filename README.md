# openai.go

> OpenAI Library in Golang


## Example

### Completion API with GPT-3 Model

```go
package main

import (
    "log"

	"github.com/song940/openai-go/openai"
)

func main() {
	config := openai.Configuration{
		APIKey: "your api key",
	}
	client, err := openai.NewClient(config)
	if err != nil {
		log.Fatal(err)
	}
	req := openai.CompletionRequest{
		Model:     openai.GPT3TextDavinci003,
		MaxTokens: 2048,
		Prompt:    "Say this is a test",
	}
	resp, err := client.CreateCompletion(req)
	if err != nil {
		panic(err)
	}
	log.Println(resp.Choices[0].Text)
}
```

### Chat API with ChatAGPT Model

```go
package main

import (
    "log"

	"github.com/song940/openai-go/openai"
)

func main() {
	config := openai.Configuration{
		APIKey: "your api key",
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
		Model:     "gpt-3.5-turbo",
		MaxTokens: 2048,
		Messages: []openai.ChatCompletionMessage{
			message,
		},
	}
	resp, err := client.CreateChatCompletion(req)
	if err != nil {
		panic(err)
	}
	log.Println(resp.Choices[0].Message.Role)
	log.Println(resp.Choices[0].Message.Content)
}
```

## License

This Project is licensed under the MIT License.