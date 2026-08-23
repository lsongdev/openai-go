package protocols

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lsongdev/miya-agents/anthropic"
	"github.com/lsongdev/miya-agents/openai"
	"github.com/lsongdev/miya-agents/proxy/codec"
	"github.com/lsongdev/miya-agents/proxy/providers"
	"github.com/lsongdev/miya-agents/proxy/providers/codex"
)

const maxObservedResponseBytes = 64 << 20

func (env *Env) forwardNative(ctx *providers.RequestContext, protocol providers.Protocol, body []byte, w http.ResponseWriter) (*openai.ChatCompletionResponse, error) {
	req, err := ctx.Upstream.NewNativeRequest(ctx.Request.Context(), protocol, body)
	if err != nil {
		WriteErrorForProtocol(w, protocol, http.StatusBadGateway, "upstream request failed: "+err.Error())
		return nil, err
	}
	copyForwardHeaders(req.Header, ctx.Request.Header)

	resp, err := env.HTTPClient.Do(req)
	if err != nil {
		WriteErrorForProtocol(w, protocol, http.StatusBadGateway, "upstream request failed: "+err.Error())
		return nil, err
	}
	defer resp.Body.Close()

	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	var observed limitedBuffer
	observed.remaining = maxObservedResponseBytes
	if err := copyResponseBody(w, io.TeeReader(resp.Body, &observed)); err != nil {
		return nil, fmt.Errorf("copy upstream response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("upstream returned %s", resp.Status)
	}
	if observed.truncated {
		return nil, nil
	}
	output, err := observeNativeResponse(protocol, ctx.Input.Stream, observed.Bytes())
	if err != nil {
		return nil, fmt.Errorf("observe native response: %w", err)
	}
	return output, nil
}

type limitedBuffer struct {
	bytes.Buffer
	remaining int
	truncated bool
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	if b.remaining > 0 {
		keep := min(len(p), b.remaining)
		_, _ = b.Buffer.Write(p[:keep])
		b.remaining -= keep
		if keep < len(p) {
			b.truncated = true
		}
	} else if len(p) > 0 {
		b.truncated = true
	}
	return n, nil
}

func observeNativeResponse(protocol providers.Protocol, stream bool, body []byte) (*openai.ChatCompletionResponse, error) {
	if !stream {
		switch protocol {
		case providers.ProtocolOpenAIChat:
			var response openai.ChatCompletionResponse
			return &response, json.Unmarshal(body, &response)
		case providers.ProtocolOpenAIResponses:
			var response openai.ResponseObject
			if err := json.Unmarshal(body, &response); err != nil {
				return nil, err
			}
			return codex.ChatCompletionFromResponseObject(&response), nil
		case providers.ProtocolAnthropic:
			var response anthropic.Response
			if err := json.Unmarshal(body, &response); err != nil {
				return nil, err
			}
			return anthropic.NewChatCompletionResponseFromAnthropicResponse(&response), nil
		}
	}
	return observeNativeStream(protocol, body)
}

func observeNativeStream(protocol providers.Protocol, body []byte) (*openai.ChatCompletionResponse, error) {
	if protocol == providers.ProtocolAnthropic {
		response, err := codec.CollectStream(context.Background(), codec.Anthropic, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		return canonicalResponseToChat(response), nil
	}
	assembler := openai.NewResponseAssembler()
	var completed *openai.ResponseObject
	scanner := bufio.NewScanner(bytes.NewReader(body))
	scanner.Buffer(make([]byte, 64*1024), maxObservedResponseBytes)
	var eventType string
	var data []string
	flush := func() error {
		if len(data) == 0 {
			eventType = ""
			return nil
		}
		payload := strings.Join(data, "\n")
		data = nil
		if payload == "[DONE]" {
			eventType = ""
			return nil
		}
		switch protocol {
		case providers.ProtocolOpenAIChat:
			var chunk openai.ChatCompletionResponse
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				return err
			}
			assembler.Update(chunk)
		case providers.ProtocolOpenAIResponses:
			if eventType == "response.completed" || eventType == "response.incomplete" || eventType == "response.failed" {
				var event struct {
					Response openai.ResponseObject `json:"response"`
				}
				if err := json.Unmarshal([]byte(payload), &event); err != nil {
					return err
				}
				completed = &event.Response
			}
		}
		eventType = ""
		return nil
	}
	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if line == "" {
			if err := flush(); err != nil {
				return nil, err
			}
			continue
		}
		if strings.HasPrefix(line, "event:") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			value := strings.TrimPrefix(line, "data:")
			data = append(data, strings.TrimPrefix(value, " "))
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if err := flush(); err != nil {
		return nil, err
	}
	if completed != nil {
		return codex.ChatCompletionFromResponseObject(completed), nil
	}
	return assembler.Build(), nil
}

func copyForwardHeaders(dst, src http.Header) {
	for _, key := range []string{"Accept", "OpenAI-Organization", "OpenAI-Project", "User-Agent"} {
		if values := src.Values(key); len(values) > 0 {
			dst.Del(key)
			for _, value := range values {
				dst.Add(key, value)
			}
		}
	}
	for _, key := range []string{"Anthropic-Beta", "OpenAI-Beta"} {
		for _, value := range src.Values(key) {
			mergeHeaderValue(dst, key, value)
		}
	}
}

func mergeHeaderValue(header http.Header, key, value string) {
	seen := map[string]bool{}
	var merged []string
	for _, source := range append(header.Values(key), value) {
		for _, part := range strings.Split(source, ",") {
			part = strings.TrimSpace(part)
			if part != "" && !seen[part] {
				seen[part] = true
				merged = append(merged, part)
			}
		}
	}
	header.Set(key, strings.Join(merged, ","))
}

func copyResponseHeaders(dst, src http.Header) {
	for key, values := range src {
		if isHopByHopHeader(key) {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func isHopByHopHeader(key string) bool {
	switch http.CanonicalHeaderKey(key) {
	case "Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization", "Te", "Trailer", "Transfer-Encoding", "Upgrade":
		return true
	default:
		return false
	}
}

func copyResponseBody(w http.ResponseWriter, body io.Reader) error {
	flusher, canFlush := w.(http.Flusher)
	buffer := make([]byte, 32*1024)
	for {
		n, readErr := body.Read(buffer)
		if n > 0 {
			if _, err := w.Write(buffer[:n]); err != nil {
				return err
			}
			if canFlush {
				flusher.Flush()
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				return nil
			}
			return readErr
		}
	}
}
