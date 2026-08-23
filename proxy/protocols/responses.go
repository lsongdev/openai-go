package protocols

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

// Responses serves the OpenAI /v1/responses API endpoint.
func (env *Env) Responses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	body, ok := readRequestBody(w, r, providers.ProtocolOpenAIResponses)
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
	chatReq := requestForHooks(providers.ProtocolOpenAIResponses, body, routing.Model, routing.Stream)
	ctx := &providers.RequestContext{
		RequestID:   env.NextRequestID(),
		Request:     r,
		Response:    w,
		Input:       chatReq,
		RawInput:    body,
		InputFormat: providers.ProtocolOpenAIResponses,
	}

	start := time.Now()

	if !env.begin(ctx, w) {
		return
	}
	if !env.resolveUpstream(ctx, routing.Model, w) {
		return
	}
	if ctx.Upstream.NativeProtocol() == providers.ProtocolOpenAIResponses {
		output, err := env.forwardNative(ctx, providers.ProtocolOpenAIResponses, body, w)
		env.end(ctx, r, output, err, start)
		return
	}
	chatResp, respErr := env.forwardTranslated(ctx, providers.ProtocolOpenAIResponses, body, w)
	env.end(ctx, r, chatResp, respErr, start)
}
