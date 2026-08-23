package main

import (
	"log"
	"net/http"
	"os"

	"github.com/lsongdev/miya-agents/proxy"
)

func main() {
	r := proxy.NewProxy()

	openaiProvider := &proxy.Provider{
		Name:    "openai",
		Type:    proxy.ProviderTypeOpenAI,
		BaseURL: "https://api.openai.com",
		APIKey:  os.Getenv("OPENAI_API_KEY"),
		Models:  []string{"gpt-4-turbo", "gpt-4o", "deepseek-chat"},
	}
	r.AddProvider(openaiProvider)

	anthropicProvider := &proxy.Provider{
		Name:             "anthropic",
		Type:             proxy.ProviderTypeAnthropic,
		BaseURL:          "https://api.anthropic.com",
		APIKey:           os.Getenv("ANTHROPIC_API_KEY"),
		DefaultMaxTokens: 4096,
		Models:           []string{"claude-3-7-sonnet-20250219"},
	}
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
