package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// ResponseWriter writes Anthropic-format SSE events to an http.ResponseWriter.
// Events are written as:
//
//	event: <type>
//	data: <json>
type ResponseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func NewResponseWriter(w http.ResponseWriter) *ResponseWriter {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	s := &ResponseWriter{w: w}
	if f, ok := w.(http.Flusher); ok {
		s.flusher = f
	}
	return s
}

func (s *ResponseWriter) Send(event Event) {
	data, _ := json.Marshal(event)
	fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event.Type, data)
	if s.flusher != nil {
		s.flusher.Flush()
	}
}

func (s *ResponseWriter) SendMessageStart(id, typ, role, model string) {
	s.Send(Event{
		Type: "message_start",
		Message: &MessageStart{
			ID:    id,
			Type:  typ,
			Role:  role,
			Model: model,
		},
	})
}

func (s *ResponseWriter) SendContentBlockStart(index int, blockType string) {
	s.Send(Event{
		Type:         "content_block_start",
		Index:        &index,
		ContentBlock: &ContentBlock{Type: blockType},
	})
}

func (s *ResponseWriter) SendContentBlockDelta(index int, delta any) {
	deltaBytes, _ := json.Marshal(delta)
	s.Send(Event{
		Type:  "content_block_delta",
		Index: &index,
		Delta: deltaBytes,
	})
}

func (s *ResponseWriter) SendContentBlockStop(index int) {
	s.Send(Event{
		Type:  "content_block_stop",
		Index: &index,
	})
}

func (s *ResponseWriter) SendMessageDelta(stopReason string, stopSequence *string, usage *Usage) {
	deltaData, _ := json.Marshal(map[string]any{
		"stop_reason":   stopReason,
		"stop_sequence": stopSequence,
	})
	s.Send(Event{
		Type:  "message_delta",
		Delta: deltaData,
		Usage: usage,
	})
}

func (s *ResponseWriter) SendMessageStop() {
	s.Send(Event{Type: "message_stop"})
}

// Done is a no-op for Anthropic SSE. The stream ends when the connection closes.
func (s *ResponseWriter) Done() {}
