package main

import (
	"log"

	"github.com/lsongdev/openai-go/mcp"
)

func main() {
	// Create MCP client
	client, err := mcp.NewStdioClient("npx", "-y", "12306-mcp")
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Initialize connection
	log.Println("Initializing connection...")
	initResult, err := client.Initialize("openai-go-mcp-example", "1.0.0")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Connected to: %s v%s (protocol: %s)",
		initResult.ServerInfo.Name,
		initResult.ServerInfo.Version,
		initResult.ProtocolVersion,
	)
	if initResult.Capabilities.Tools != nil {
		log.Printf("- Tools support: yes (listChanged: %v)", initResult.Capabilities.Tools.ListChanged)
	} else {
		log.Println("- Tools support: no")
	}
	if initResult.Capabilities.Prompts != nil {
		log.Printf("- Prompts support: yes (listChanged: %v)", initResult.Capabilities.Prompts.ListChanged)
	} else {
		log.Println("- Prompts support: no")
	}

	// List available tools
	log.Println("Listing available tools...")
	tools, err := client.ListTools()
	if err != nil {
		log.Printf("Note: %v", err)
		log.Println("This server may not support the 'tools/list' method.")
	} else {
		if len(tools) == 0 {
			log.Println("No tools available")
		} else {
			log.Printf("Found %d tool(s):\n", len(tools))
			for _, tool := range tools {
				log.Printf("- %s: %s", tool.Name, tool.Description)
			}
		}
	}
	result, err := client.CallTool("get-current-date", map[string]any{})
	log.Println("get-current-date", result.Content[0].Text)

}
