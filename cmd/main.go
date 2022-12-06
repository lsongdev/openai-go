package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/lsongdev/openai-go/openai"
	"github.com/lsongdev/openai-go/tools"
)

var (
	api    string
	apiKey string
	model  string
)

func main() {
	flag.StringVar(&api, "api", os.Getenv("OPENAI_API"), "OpenAI API endpoint")
	flag.StringVar(&apiKey, "api-key", os.Getenv("OPENAI_API_KEY"), "OpenAI API key (required)")
	flag.StringVar(&model, "model", openai.DeepSeekChat, "Model to use")
	flag.Parse()

	if apiKey == "" {
		log.Fatal("API key is required. Use --api-key flag or set OPENAI_API_KEY environment variable")
	}

	args := flag.Args()
	var command string
	if len(args) > 0 {
		command = args[0]
	}

	client, err := openai.NewClient(&openai.Configuration{
		API:    api,
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatal(err)
	}

	switch command {
	case "models":
		listModels(client)
	case "send":
		if len(args) < 2 {
			log.Fatal("Usage: send <message>")
		}
		sendMessage(client, strings.Join(args[1:], " "))
	case "chat":
		startChat(client)
	default:
		fmt.Println("Usage: cmd <command> [arguments]")
		fmt.Println("Commands:")
		fmt.Println("  models              List available models")
		fmt.Println("  send <message>      Send a single message")
		fmt.Println("  chat                Start an interactive chat session")
	}
}

func listModels(client *openai.OpenAIClient) {
	models, err := client.Models()
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range models {
		fmt.Println(m.ID, m.Object, m.OwnedBy)
	}
}

func sendMessage(client *openai.OpenAIClient, message string) {
	workspace, _ := os.Getwd()
	toolsMap := createTools(workspace)

	messages := []openai.ChatCompletionMessage{
		openai.UserMessage(message),
	}

	runChatLoop(client, &messages, toolsMap)
}

func startChat(client *openai.OpenAIClient) {
	workspace, _ := os.Getwd()
	toolsMap := createTools(workspace)

	messages := []openai.ChatCompletionMessage{
		openai.SystemMessage("You are a helpful assistant with access to shell, file, and web tools."),
	}

	fmt.Println("Interactive chat started. Type 'quit' or 'exit' to end.")
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("You: ")
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}
		if input == "quit" || input == "exit" {
			break
		}

		messages = append(messages, openai.UserMessage(input))
		runChatLoop(client, &messages, toolsMap)
	}
}

func createTools(workspace string) map[string]openai.Tool {
	return map[string]openai.Tool{
		"read_file":   &tools.ReadFileTool{Workspace: workspace},
		"write_file":  &tools.WriteFileTool{Workspace: workspace},
		"append_file": &tools.AppendFileTool{Workspace: workspace},
		"edit_file":   &tools.EditFileTool{Workspace: workspace},
		"exec": &tools.ExecTool{
			Workspace:           workspace,
			DefaultTimeout:      tools.ExecDefaultTimeoutSeconds,
			RestrictToWorkspace: true,
		},
	}
}

func runChatLoop(client *openai.OpenAIClient, messages *[]openai.ChatCompletionMessage, toolsMap map[string]openai.Tool) {
	for {
		// Build tools list from toolsMap
		var tools []openai.ToolDef
		for _, tool := range toolsMap {
			tools = append(tools, tool.Def())
		}
		req := openai.ChatCompletionRequest{
			Model:    model,
			Messages: *messages,
			Tools:    tools,
			Stream:   true,
		}

		var respMessage openai.ChatCompletionMessage
		if req.Stream {
			resp, err := client.CreateChatCompletionStream(&req)
			if err != nil {
				log.Fatal(err)
			}
			builder := openai.CreateMessageBuilder()
			for chunk := range resp {
				m := chunk.GetMessage()
				builder.Update(*m)
				fmt.Print(m.Content)
			}
			fmt.Println()
			respMessage = builder.Build()
		} else {
			resp, err := client.CreateChatCompletion(&req)
			if err != nil {
				log.Fatal(err)
			}
			respMessage = *resp.GetMessage()
			fmt.Println(respMessage.Content)
		}
		*messages = append(*messages, respMessage)

		if !respMessage.HasToolCall() {
			return
		}

		// Execute tool calls
		for _, toolCall := range respMessage.ToolCalls {
			tool, ok := toolsMap[toolCall.Function.Name]
			var result string
			if ok {
				result = tool.Run(context.Background(), toolCall.Function.Arguments)
			} else {
				result = fmt.Sprintf("Error: unknown tool '%s'", toolCall.Function.Name)
			}
			*messages = append(*messages, openai.ToolResultMessage(toolCall.ID, toolCall.Function.Name, result))
		}
	}
}
