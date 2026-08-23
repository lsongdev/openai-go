package protocols

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy/providers"
)

// Embeddings serves the OpenAI /v1/embeddings API endpoint.
func (env *Env) Embeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		WriteError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var embReq openai.EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&embReq); err != nil {
		WriteError(w, http.StatusBadRequest, fmt.Sprintf("invalid request body: %v", err))
		return
	}
	ctx := &providers.RequestContext{
		RequestID: env.NextRequestID(),
		Request:   r,
		Response:  w,
		Input:     &openai.ChatCompletionRequest{Model: embReq.Model},
	}
	start := time.Now()
	if !env.begin(ctx, w) {
		return
	}
	if !env.resolveUpstream(ctx, embReq.Model, w) {
		return
	}
	client := providers.NewOpenAIClient(ctx.Upstream, env.HTTPClient)
	resp, err := client.CreateEmbeddings(ctx.Request.Context(), &embReq)
	if err != nil {
		WriteError(w, http.StatusBadGateway, fmt.Sprintf("upstream request failed: %v", err))
		return
	}
	writeJSON(w, resp)

	env.end(ctx, r, nil, nil, start)
}
