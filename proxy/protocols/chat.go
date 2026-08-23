package protocols

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

// ChatCompletions serves the OpenAI /v1/chat/completions API endpoint.
func (env *Env) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	body, ok := readRequestBody(w, r, providers.ProtocolOpenAIChat)
	if !ok {
		return
	}
	var routing struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &routing); err != nil {
		WriteError(w, http.StatusBadRequest, fmt.Sprintf("invalid request body: %v", err))
		return
	}
	chatReq := requestForHooks(providers.ProtocolOpenAIChat, body, routing.Model, routing.Stream)

	ctx := &providers.RequestContext{
		RequestID:   env.NextRequestID(),
		Request:     r,
		Response:    w,
		Input:       chatReq,
		RawInput:    body,
		InputFormat: providers.ProtocolOpenAIChat,
	}

	start := time.Now()

	if !env.begin(ctx, w) {
		return
	}
	if !env.resolveUpstream(ctx, routing.Model, w) {
		return
	}
	if ctx.Upstream.NativeProtocol() == providers.ProtocolOpenAIChat {
		output, err := env.forwardNative(ctx, providers.ProtocolOpenAIChat, body, w)
		env.end(ctx, r, output, err, start)
		return
	}
	chatResp, respErr := env.forwardTranslated(ctx, providers.ProtocolOpenAIChat, body, w)
	env.end(ctx, r, chatResp, respErr, start)
}
