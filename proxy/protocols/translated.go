package protocols

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy/codec"
	"github.com/lsongdev/miya-agents/proxy/providers"
)

const maxTranslatedResponseBytes = 64 << 20

func (env *Env) forwardTranslated(ctx *providers.RequestContext, clientProtocol providers.Protocol, body []byte, w http.ResponseWriter) (*openai.ChatCompletionResponse, error) {
	upstreamProtocol := ctx.Upstream.NativeProtocol()
	upstreamBody, canonicalRequest, diagnostics, err := codec.TranslateRequest(
		codec.Protocol(clientProtocol), codec.Protocol(upstreamProtocol), body,
	)
	appendDiagnostics(ctx, diagnostics)
	if err != nil {
		WriteErrorForProtocol(w, clientProtocol, http.StatusBadRequest, err.Error())
		return nil, err
	}
	req, err := ctx.Upstream.NewTranslatedRequest(ctx.Request.Context(), upstreamProtocol, upstreamBody)
	if err != nil {
		WriteErrorForProtocol(w, clientProtocol, http.StatusBadGateway, "upstream request failed: "+err.Error())
		return nil, err
	}
	copyForwardHeaders(req.Header, ctx.Request.Header)
	resp, err := env.HTTPClient.Do(req)
	if err != nil {
		WriteErrorForProtocol(w, clientProtocol, http.StatusBadGateway, "upstream request failed: "+err.Error())
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		upstreamError, readErr := io.ReadAll(io.LimitReader(resp.Body, maxTranslatedResponseBytes))
		if readErr != nil {
			upstreamError = []byte(readErr.Error())
		}
		err = fmt.Errorf("upstream returned %s: %s", resp.Status, string(upstreamError))
		WriteErrorForProtocol(w, clientProtocol, resp.StatusCode, err.Error())
		return nil, err
	}
	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	upstreamIsStream := strings.Contains(contentType, "text/event-stream") || (contentType == "" && ctx.Upstream.AlwaysStream)
	if canonicalRequest.Stream && upstreamIsStream {
		response, err := codec.TranslateStream(
			ctx.Request.Context(), codec.Protocol(upstreamProtocol), codec.Protocol(clientProtocol), resp.Body, w,
		)
		if err != nil {
			return nil, fmt.Errorf("translate upstream stream: %w", err)
		}
		return canonicalResponseToChat(response), nil
	}
	if upstreamIsStream {
		response, err := codec.CollectStream(ctx.Request.Context(), codec.Protocol(upstreamProtocol), resp.Body)
		if err != nil {
			WriteErrorForProtocol(w, clientProtocol, http.StatusBadGateway, "translate upstream stream: "+err.Error())
			return nil, err
		}
		clientBody, diagnostics, err := codec.EncodeResponse(codec.Protocol(clientProtocol), response)
		appendDiagnostics(ctx, diagnostics)
		if blocking := codec.BlockingDiagnostics(diagnostics); err != nil || len(blocking) > 0 {
			if err == nil {
				err = &codec.TranslationError{From: codec.Protocol(upstreamProtocol), To: codec.Protocol(clientProtocol), Diagnostics: blocking}
			}
			WriteErrorForProtocol(w, clientProtocol, http.StatusBadGateway, "translate upstream response: "+err.Error())
			return nil, err
		}
		if clientProtocol == providers.ProtocolOpenAIResponses {
			clientBody = enrichResponsesBody(clientBody, body)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(clientBody)
		return canonicalResponseToChat(response), nil
	}
	upstreamResponse, err := io.ReadAll(io.LimitReader(resp.Body, maxTranslatedResponseBytes+1))
	if err != nil {
		WriteErrorForProtocol(w, clientProtocol, http.StatusBadGateway, "read upstream response: "+err.Error())
		return nil, err
	}
	if len(upstreamResponse) > maxTranslatedResponseBytes {
		err = fmt.Errorf("upstream response exceeds %d bytes", maxTranslatedResponseBytes)
		WriteErrorForProtocol(w, clientProtocol, http.StatusBadGateway, err.Error())
		return nil, err
	}
	response, diagnostics, err := codec.DecodeResponse(codec.Protocol(upstreamProtocol), upstreamResponse)
	if err != nil {
		WriteErrorForProtocol(w, clientProtocol, http.StatusBadGateway, "translate upstream response: "+err.Error())
		return nil, err
	}
	appendDiagnostics(ctx, diagnostics)
	if blocking := codec.BlockingDiagnostics(diagnostics); len(blocking) > 0 {
		err = &codec.TranslationError{From: codec.Protocol(upstreamProtocol), To: codec.Protocol(clientProtocol), Diagnostics: blocking}
		WriteErrorForProtocol(w, clientProtocol, http.StatusBadGateway, "translate upstream response: "+err.Error())
		return nil, err
	}
	if canonicalRequest.Stream {
		if err := codec.WriteResponseStream(codec.Protocol(clientProtocol), response, w); err != nil {
			return nil, fmt.Errorf("render client stream: %w", err)
		}
		return canonicalResponseToChat(response), nil
	}
	clientBody, diagnostics, err := codec.EncodeResponse(codec.Protocol(clientProtocol), response)
	appendDiagnostics(ctx, diagnostics)
	if blocking := codec.BlockingDiagnostics(diagnostics); err != nil || len(blocking) > 0 {
		if err == nil {
			err = &codec.TranslationError{From: codec.Protocol(upstreamProtocol), To: codec.Protocol(clientProtocol), Diagnostics: blocking}
		}
		WriteErrorForProtocol(w, clientProtocol, http.StatusBadGateway, "translate upstream response: "+err.Error())
		return nil, err
	}
	if clientProtocol == providers.ProtocolOpenAIResponses {
		clientBody = enrichResponsesBody(clientBody, body)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(clientBody)
	return canonicalResponseToChat(response), nil
}

func appendDiagnostics(ctx *providers.RequestContext, diagnostics []codec.Diagnostic) {
	for _, diagnostic := range diagnostics {
		severity := diagnostic.Severity
		if severity == "" {
			severity = "error"
		}
		ctx.Diagnostics = append(ctx.Diagnostics, fmt.Sprintf("[%s] %s: %s", severity, diagnostic.Path, diagnostic.Message))
	}
}

func enrichResponsesBody(responseBody, requestBody []byte) []byte {
	var response map[string]any
	var request map[string]any
	if json.Unmarshal(responseBody, &response) != nil || json.Unmarshal(requestBody, &request) != nil {
		return responseBody
	}
	for _, key := range []string{
		"instructions", "parallel_tool_calls", "previous_response_id", "temperature", "top_p",
		"max_output_tokens", "tool_choice", "tools", "metadata",
	} {
		if value, ok := request[key]; ok {
			response[key] = value
		}
	}
	enriched, err := json.Marshal(response)
	if err != nil {
		return responseBody
	}
	return enriched
}

func canonicalResponseToChat(response *codec.Response) *openai.ChatCompletionResponse {
	if response == nil {
		return nil
	}
	message := &openai.ChatCompletionMessage{Role: openai.RoleAssistant}
	for _, content := range response.Content {
		switch content.Type {
		case codec.ContentText:
			message.Content += content.Text
		case codec.ContentReasoning:
			message.ReasoningContent += content.Text
		case codec.ContentToolCall:
			arguments, _ := json.Marshal(content.Arguments)
			message.ToolCalls = append(message.ToolCalls, openai.ToolCall{
				Index: len(message.ToolCalls), ID: content.ID, Type: "function",
				Function: openai.FunctionCall{Name: content.Name, Arguments: string(arguments)},
			})
		}
	}
	result := &openai.ChatCompletionResponse{
		ID: response.ID, Object: "chat.completion", Created: response.CreatedAt, Model: response.Model,
		Choices: []openai.ChatCompletionChoice{{Index: 0, Message: message, FinishReason: canonicalFinishReason(response.StopReason)}},
	}
	if response.Usage != nil {
		result.Usage = &openai.CompletionUsage{
			PromptTokens: response.Usage.InputTokens, CompletionTokens: response.Usage.OutputTokens, TotalTokens: response.Usage.TotalTokens,
		}
		result.Usage.PromptTokensDetails.CachedTokens = response.Usage.CachedInputTokens
		result.Usage.CompletionTokensDetails.ReasoningTokens = response.Usage.ReasoningTokens
	}
	return result
}

func canonicalFinishReason(reason string) string {
	switch reason {
	case "length":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "content_filter":
		return "content_filter"
	default:
		return "stop"
	}
}
