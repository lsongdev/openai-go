package protocols

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy/codec"
	"github.com/lsongdev/miya-agents/proxy/providers"
)

const maxRequestBodyBytes = 32 << 20

func readRequestBody(w http.ResponseWriter, r *http.Request, protocol providers.Protocol) ([]byte, bool) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBodyBytes))
	if err != nil {
		WriteErrorForProtocol(w, protocol, http.StatusBadRequest, fmt.Sprintf("invalid request body: %v", err))
		return nil, false
	}
	return body, true
}

// Env carries the facilities protocol handlers need from the proxy core.
type Env struct {
	// FindProvider resolves a model name to its upstream provider.
	FindProvider func(model string) *providers.Provider
	// ListProviders returns all registered providers.
	ListProviders func() []*providers.Provider
	// HTTPClient is the shared upstream HTTP client.
	HTTPClient *http.Client
	// NextRequestID mints unique request IDs.
	NextRequestID func() string
	// OnRequest is invoked before dispatching; returning an error rejects
	// the request. It may set ctx.Upstream to override provider selection.
	OnRequest func(*providers.RequestContext) error
	// OnResponse is invoked after the response has been written.
	OnResponse func(*providers.ResponseContext)
}

// WriteError renders an error payload in OpenAI error format.
func WriteError(w http.ResponseWriter, status int, message string) {
	WriteErrorForProtocol(w, providers.ProtocolOpenAIChat, status, message)
}

func WriteErrorForProtocol(w http.ResponseWriter, protocol providers.Protocol, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if protocol == providers.ProtocolAnthropic {
		fmt.Fprintf(w, `{"type":"error","error":{"type":"invalid_request_error","message":%q}}`, message)
		return
	}
	fmt.Fprintf(w, `{"error":{"message":%q,"type":"invalid_request_error"}}`, message)
}

// begin runs the OnRequest hook; it returns false when the request was
// rejected and an error response has already been written.
func (env *Env) begin(ctx *providers.RequestContext, w http.ResponseWriter) bool {
	if env.OnRequest == nil {
		return true
	}
	if err := env.OnRequest(ctx); err != nil {
		if reqErr, ok := err.(*providers.RequestError); ok {
			WriteErrorForProtocol(w, ctx.InputFormat, reqErr.Status, reqErr.Message)
		} else {
			WriteErrorForProtocol(w, ctx.InputFormat, http.StatusForbidden, fmt.Sprintf("request rejected: %v", err))
		}
		return false
	}
	return true
}

// resolveUpstream picks the upstream provider for the model when the hook did
// not set one; it returns false when no provider is available.
func (env *Env) resolveUpstream(ctx *providers.RequestContext, model string, w http.ResponseWriter) bool {
	if ctx.Upstream == nil && env.FindProvider != nil {
		ctx.Upstream = env.FindProvider(model)
	}
	if ctx.Upstream == nil {
		WriteErrorForProtocol(w, ctx.InputFormat, http.StatusBadRequest, fmt.Sprintf("no provider available for model %s", model))
		return false
	}
	return true
}

// end invokes the OnResponse hook.
func (env *Env) end(ctx *providers.RequestContext, r *http.Request, chatResp *openai.ChatCompletionResponse, respErr error, start time.Time) {
	if env.OnResponse == nil {
		return
	}
	env.OnResponse(&providers.ResponseContext{
		RequestID:   ctx.RequestID,
		Request:     r,
		Response:    ctx.Response,
		Input:       ctx.Input,
		Output:      chatResp,
		Error:       respErr,
		Duration:    time.Since(start),
		Diagnostics: append([]string(nil), ctx.Diagnostics...),
	})
}

// writeJSON writes a JSON payload with 200 OK.
func writeJSON(w http.ResponseWriter, v any) {
	out, _ := json.Marshal(v)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(out)
}

func requestForHooks(protocol providers.Protocol, body []byte, model string, stream bool) *openai.ChatCompletionRequest {
	request := &openai.ChatCompletionRequest{Model: model, Stream: stream}
	canonical, _, err := codec.DecodeRequest(codec.Protocol(protocol), body)
	if err != nil || canonical == nil {
		return request
	}
	request.MaxTokens = canonical.MaxOutputTokens
	request.Temperature = canonical.Temperature
	request.TopP = canonical.TopP
	request.Stop = canonical.Stop
	for _, message := range canonical.Messages {
		role := message.Role
		if role == "developer" {
			role = openai.RoleSystem
		}
		converted := openai.ChatCompletionMessage{Role: role}
		for _, content := range message.Content {
			switch content.Type {
			case codec.ContentText:
				converted.Content += content.Text
			case codec.ContentToolCall:
				arguments, _ := json.Marshal(content.Arguments)
				converted.ToolCalls = append(converted.ToolCalls, openai.ToolCall{
					Index: len(converted.ToolCalls), ID: content.ID, Type: "function",
					Function: openai.FunctionCall{Name: content.Name, Arguments: string(arguments)},
				})
			case codec.ContentToolResult:
				request.Messages = append(request.Messages, openai.ChatCompletionMessage{
					Role: openai.RoleTool, ToolCallID: content.ToolCallID, Content: content.Text,
				})
			}
		}
		if converted.Content != "" || len(converted.ToolCalls) > 0 || len(message.Content) == 0 {
			request.Messages = append(request.Messages, converted)
		}
	}
	for _, tool := range canonical.Tools {
		request.Tools = append(request.Tools, openai.ToolDef{Type: "function", Function: openai.FunctionDef{
			Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema,
		}})
	}
	return request
}
