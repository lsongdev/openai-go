package protocols

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/lsongdev/miya-agents/proxy/providers"
)

// Messages serves the Anthropic /v1/messages API endpoint.
func (env *Env) Messages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteErrorForProtocol(w, providers.ProtocolAnthropic, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	body, ok := readRequestBody(w, r, providers.ProtocolAnthropic)
	if !ok {
		return
	}
	var routing struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &routing); err != nil {
		WriteErrorForProtocol(w, providers.ProtocolAnthropic, http.StatusBadRequest, fmt.Sprintf("invalid request body: %v", err))
		return
	}
	chatReq := requestForHooks(providers.ProtocolAnthropic, body, routing.Model, routing.Stream)
	ctx := &providers.RequestContext{
		RequestID:   env.NextRequestID(),
		Request:     r,
		Response:    w,
		Input:       chatReq,
		RawInput:    body,
		InputFormat: providers.ProtocolAnthropic,
	}

	start := time.Now()

	if !env.begin(ctx, w) {
		return
	}
	if !env.resolveUpstream(ctx, routing.Model, w) {
		return
	}
	if ctx.Upstream.NativeProtocol() == providers.ProtocolAnthropic {
		output, err := env.forwardNative(ctx, providers.ProtocolAnthropic, body, w)
		env.end(ctx, r, output, err, start)
		return
	}
	chatResp, respErr := env.forwardTranslated(ctx, providers.ProtocolAnthropic, body, w)
	env.end(ctx, r, chatResp, respErr, start)
}
