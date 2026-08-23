package main

import (
	"log"
	"net/http"
	"os"

	"github.com/lsongdev/miya-agents/proxy"
	anthropicprovider "github.com/lsongdev/miya-agents/proxy/providers/anthropic"
	openaiprovider "github.com/lsongdev/miya-agents/proxy/providers/openai"
)

func main() {
	r := proxy.NewProxy()

	openaiProvider := openaiprovider.Provider("openai", "https://api.openai.com", os.Getenv("OPENAI_API_KEY"))
	openaiProvider.Models = []string{"gpt-4-turbo", "gpt-4o", "deepseek-chat"}
	r.AddProvider(openaiProvider)

	anthropicProvider := anthropicprovider.Provider("anthropic", "https://api.anthropic.com", os.Getenv("ANTHROPIC_API_KEY"))
	anthropicProvider.DefaultMaxTokens = 4096
	anthropicProvider.Models = []string{"claude-3-7-sonnet-20250219"}
	r.AddProvider(anthropicProvider)

	r.OnRequest(func(ctx *proxy.RequestContext) error {
		log.Printf("[REQUEST] %s model=%s stream=%v", ctx.RequestID, ctx.Input.Model, ctx.Input.Stream)
		return nil
	})

	addr := ":8080"
	if v := os.Getenv("ADDR"); v != "" {
		addr = v
	}
	log.Printf("Proxy listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, r))
}
