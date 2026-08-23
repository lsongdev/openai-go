package codec

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lsongdev/miya-agents/sse"
)

type streamEventType string

const (
	streamStart        streamEventType = "start"
	streamContentStart streamEventType = "content_start"
	streamContentDelta streamEventType = "content_delta"
	streamContentStop  streamEventType = "content_stop"
	streamDone         streamEventType = "done"
)

type streamEvent struct {
	Type           streamEventType
	ID             string
	Model          string
	Index          int
	Content        Content
	ArgumentsDelta string
	StopReason     string
	Usage          *Usage
}

// TranslateStream converts one provider's SSE stream to the protocol expected
// by the client while preserving text, tool calls, incremental JSON and usage.
func TranslateStream(ctx context.Context, from, to Protocol, source io.Reader, w http.ResponseWriter) (*Response, error) {
	decoder, err := newStreamDecoder(from)
	if err != nil {
		return nil, err
	}
	encoder, err := newStreamEncoder(to, w)
	if err != nil {
		return nil, err
	}
	if err := sse.Read(ctx, source, func(event sse.Event) error {
		canonical, err := decoder.Decode(event)
		if err != nil {
			return fmt.Errorf("decode %s stream event %q: %w", from, event.Type, err)
		}
		for _, item := range canonical {
			if err := encoder.Encode(item); err != nil {
				return fmt.Errorf("encode %s stream event: %w", to, err)
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}
	for _, item := range decoder.Finish() {
		if err := encoder.Encode(item); err != nil {
			return nil, err
		}
	}
	return encoder.Response(), nil
}

// CollectStream consumes a provider stream into one canonical response. It is
// used when an upstream only supports streaming but the client requested JSON.
func CollectStream(ctx context.Context, protocol Protocol, source io.Reader) (*Response, error) {
	decoder, err := newStreamDecoder(protocol)
	if err != nil {
		return nil, err
	}
	collector := streamEncoderBase{response: Response{}, blocks: map[int]*streamBlock{}}
	if err := sse.Read(ctx, source, func(event sse.Event) error {
		canonical, err := decoder.Decode(event)
		if err != nil {
			return err
		}
		for _, item := range canonical {
			collector.apply(item)
		}
		return nil
	}); err != nil {
		return nil, err
	}
	for _, item := range decoder.Finish() {
		collector.apply(item)
	}
	return collector.Response(), nil
}

// WriteResponseStream renders a complete canonical response as a valid stream.
// It provides a protocol-correct fallback for providers that ignore stream=true.
func WriteResponseStream(protocol Protocol, response *Response, w http.ResponseWriter) error {
	encoder, err := newStreamEncoder(protocol, w)
	if err != nil {
		return err
	}
	if err := encoder.Encode(streamEvent{Type: streamStart, ID: response.ID, Model: response.Model, Usage: response.Usage}); err != nil {
		return err
	}
	for index, content := range response.Content {
		startContent := content
		startContent.Text = ""
		if content.Type == ContentToolCall {
			startContent.Arguments = map[string]any{}
		}
		if err := encoder.Encode(streamEvent{Type: streamContentStart, Index: index, Content: startContent}); err != nil {
			return err
		}
		delta := streamEvent{Type: streamContentDelta, Index: index, Content: content}
		if content.Type == ContentToolCall {
			arguments, _ := json.Marshal(content.Arguments)
			delta.ArgumentsDelta = string(arguments)
			delta.Content.Arguments = nil
		}
		if err := encoder.Encode(delta); err != nil {
			return err
		}
		if err := encoder.Encode(streamEvent{Type: streamContentStop, Index: index}); err != nil {
			return err
		}
	}
	return encoder.Encode(streamEvent{Type: streamDone, StopReason: response.StopReason, Usage: response.Usage})
}

type streamDecoder interface {
	Decode(sse.Event) ([]streamEvent, error)
	Finish() []streamEvent
}

func newStreamDecoder(protocol Protocol) (streamDecoder, error) {
	switch protocol {
	case OpenAIResponses:
		return &responsesStreamDecoder{blocks: map[string]int{}, open: map[int]bool{}}, nil
	case Anthropic:
		return &anthropicStreamDecoder{open: map[int]ContentType{}}, nil
	case OpenAIChat:
		return &chatStreamDecoder{toolIndexes: map[int]int{}, open: map[int]bool{}}, nil
	default:
		return nil, fmt.Errorf("unsupported stream protocol %q", protocol)
	}
}

type responsesStreamDecoder struct {
	started   bool
	done      bool
	nextBlock int
	blocks    map[string]int
	open      map[int]bool
	hadTool   bool
	response  Response
}

func (d *responsesStreamDecoder) Decode(event sse.Event) ([]streamEvent, error) {
	if event.Data == "[DONE]" {
		return nil, nil
	}
	var value map[string]any
	if err := json.Unmarshal([]byte(event.Data), &value); err != nil {
		return nil, err
	}
	typeName := event.Type
	if typeName == "" {
		typeName, _ = stringValue(value["type"])
	}
	switch typeName {
	case "response.created", "response.in_progress":
		response, _ := object(value["response"])
		d.captureResponse(response)
		if d.started {
			return nil, nil
		}
		d.started = true
		return []streamEvent{{Type: streamStart, ID: d.response.ID, Model: d.response.Model}}, nil
	case "response.output_item.added":
		item, _ := object(value["item"])
		if item["type"] != "function_call" {
			return nil, nil
		}
		index := d.allocateBlock(itemKey(value, item))
		d.hadTool = true
		id, _ := stringValue(item["call_id"])
		if id == "" {
			id, _ = stringValue(item["id"])
		}
		name, _ := stringValue(item["name"])
		d.open[index] = true
		return []streamEvent{{Type: streamContentStart, Index: index, Content: Content{Type: ContentToolCall, ID: id, Name: name, Arguments: map[string]any{}}}}, nil
	case "response.content_part.added":
		part, _ := object(value["part"])
		if part["type"] != "output_text" {
			return nil, nil
		}
		index := d.allocateBlock(contentKey(value))
		d.open[index] = true
		return []streamEvent{{Type: streamContentStart, Index: index, Content: Content{Type: ContentText}}}, nil
	case "response.output_text.delta":
		index := d.allocateBlock(contentKey(value))
		delta, _ := stringValue(value["delta"])
		return []streamEvent{{Type: streamContentDelta, Index: index, Content: Content{Type: ContentText, Text: delta}}}, nil
	case "response.function_call_arguments.delta":
		index := d.allocateBlock(itemKey(value, nil))
		delta, _ := stringValue(value["delta"])
		return []streamEvent{{Type: streamContentDelta, Index: index, Content: Content{Type: ContentToolCall}, ArgumentsDelta: delta}}, nil
	case "response.content_part.done":
		return d.stopBlock(contentKey(value)), nil
	case "response.output_item.done":
		item, _ := object(value["item"])
		if item["type"] == "function_call" {
			return d.stopBlock(itemKey(value, item)), nil
		}
		return nil, nil
	case "response.completed", "response.incomplete":
		response, _ := object(value["response"])
		d.captureResponse(response)
		if d.hadTool && d.response.StopReason == "stop" {
			d.response.StopReason = "tool_use"
		}
		d.done = true
		return []streamEvent{{Type: streamDone, StopReason: d.response.StopReason, Usage: d.response.Usage}}, nil
	case "error", "response.failed":
		return nil, fmt.Errorf("upstream response stream failed: %s", event.Data)
	default:
		return nil, nil
	}
}

func (d *responsesStreamDecoder) Finish() []streamEvent {
	if d.done {
		return nil
	}
	return []streamEvent{{Type: streamDone, StopReason: "stop", Usage: d.response.Usage}}
}

func (d *responsesStreamDecoder) captureResponse(value map[string]any) {
	if value == nil {
		return
	}
	if id, ok := stringValue(value["id"]); ok {
		d.response.ID = id
	}
	if model, ok := stringValue(value["model"]); ok {
		d.response.Model = model
	}
	status, _ := stringValue(value["status"])
	d.response.StopReason = responsesStopReason(status, value["incomplete_details"])
	if usage := decodeResponsesUsage(value["usage"]); usage != nil {
		d.response.Usage = usage
	}
}

func (d *responsesStreamDecoder) allocateBlock(key string) int {
	if index, ok := d.blocks[key]; ok {
		return index
	}
	index := d.nextBlock
	d.nextBlock++
	d.blocks[key] = index
	return index
}

func (d *responsesStreamDecoder) stopBlock(key string) []streamEvent {
	index, ok := d.blocks[key]
	if !ok || !d.open[index] {
		return nil
	}
	delete(d.open, index)
	return []streamEvent{{Type: streamContentStop, Index: index}}
}

func itemKey(event, item map[string]any) string {
	if id, ok := stringValue(event["item_id"]); ok && id != "" {
		return "item:" + id
	}
	if item != nil {
		if id, ok := stringValue(item["id"]); ok && id != "" {
			return "item:" + id
		}
	}
	index, _ := intValue(event["output_index"])
	return fmt.Sprintf("output:%d", index)
}

func contentKey(event map[string]any) string {
	return fmt.Sprintf("%s:content:%v", itemKey(event, nil), event["content_index"])
}

type anthropicStreamDecoder struct {
	started  bool
	done     bool
	response Response
	open     map[int]ContentType
}

func (d *anthropicStreamDecoder) Decode(event sse.Event) ([]streamEvent, error) {
	var value map[string]any
	if err := json.Unmarshal([]byte(event.Data), &value); err != nil {
		return nil, err
	}
	typeName := event.Type
	if typeName == "" {
		typeName, _ = stringValue(value["type"])
	}
	switch typeName {
	case "message_start":
		message, _ := object(value["message"])
		d.response.ID, _ = stringValue(message["id"])
		d.response.Model, _ = stringValue(message["model"])
		d.response.Usage = decodeAnthropicUsage(message["usage"])
		d.started = true
		return []streamEvent{{Type: streamStart, ID: d.response.ID, Model: d.response.Model, Usage: d.response.Usage}}, nil
	case "content_block_start":
		index, _ := intValue(value["index"])
		block, _ := object(value["content_block"])
		switch block["type"] {
		case "text":
			d.open[index] = ContentText
			return []streamEvent{{Type: streamContentStart, Index: index, Content: Content{Type: ContentText}}}, nil
		case "tool_use":
			id, _ := stringValue(block["id"])
			name, _ := stringValue(block["name"])
			d.open[index] = ContentToolCall
			return []streamEvent{{Type: streamContentStart, Index: index, Content: Content{Type: ContentToolCall, ID: id, Name: name, Arguments: map[string]any{}}}}, nil
		case "thinking":
			d.open[index] = ContentReasoning
			return []streamEvent{{Type: streamContentStart, Index: index, Content: Content{Type: ContentReasoning}}}, nil
		case "redacted_thinking":
			return nil, nil
		default:
			return nil, fmt.Errorf("unsupported Anthropic content block %q", block["type"])
		}
	case "content_block_delta":
		index, _ := intValue(value["index"])
		delta, _ := object(value["delta"])
		switch delta["type"] {
		case "text_delta":
			text, _ := stringValue(delta["text"])
			var result []streamEvent
			if _, ok := d.open[index]; !ok {
				d.open[index] = ContentText
				result = append(result, streamEvent{Type: streamContentStart, Index: index, Content: Content{Type: ContentText}})
			}
			return append(result, streamEvent{Type: streamContentDelta, Index: index, Content: Content{Type: ContentText, Text: text}}), nil
		case "input_json_delta":
			partial, _ := stringValue(delta["partial_json"])
			return []streamEvent{{Type: streamContentDelta, Index: index, Content: Content{Type: ContentToolCall}, ArgumentsDelta: partial}}, nil
		case "thinking_delta":
			thinking, _ := stringValue(delta["thinking"])
			var result []streamEvent
			if _, ok := d.open[index]; !ok {
				d.open[index] = ContentReasoning
				result = append(result, streamEvent{Type: streamContentStart, Index: index, Content: Content{Type: ContentReasoning}})
			}
			return append(result, streamEvent{Type: streamContentDelta, Index: index, Content: Content{Type: ContentReasoning, Text: thinking}}), nil
		case "signature_delta":
			return nil, nil
		default:
			return nil, nil
		}
	case "content_block_stop":
		index, _ := intValue(value["index"])
		delete(d.open, index)
		return []streamEvent{{Type: streamContentStop, Index: index}}, nil
	case "message_delta":
		delta, _ := object(value["delta"])
		stopReason, _ := stringValue(delta["stop_reason"])
		d.response.StopReason = anthropicStopReason(stopReason)
		if usage := decodeAnthropicUsage(value["usage"]); usage != nil {
			if d.response.Usage != nil {
				usage.InputTokens = d.response.Usage.InputTokens
				usage.CachedInputTokens = d.response.Usage.CachedInputTokens
			}
			usage.TotalTokens = usage.InputTokens + usage.OutputTokens
			d.response.Usage = usage
		}
		return nil, nil
	case "message_stop":
		d.done = true
		var result []streamEvent
		for index := range d.open {
			result = append(result, streamEvent{Type: streamContentStop, Index: index})
		}
		d.open = map[int]ContentType{}
		return append(result, streamEvent{Type: streamDone, StopReason: d.response.StopReason, Usage: d.response.Usage}), nil
	case "error":
		return nil, fmt.Errorf("upstream Anthropic stream failed: %s", event.Data)
	default:
		return nil, nil
	}
}

func (d *anthropicStreamDecoder) Finish() []streamEvent {
	if d.done {
		return nil
	}
	return []streamEvent{{Type: streamDone, StopReason: "stop", Usage: d.response.Usage}}
}

type chatStreamDecoder struct {
	started     bool
	done        bool
	nextBlock   int
	textIndex   int
	textOpen    bool
	toolIndexes map[int]int
	open        map[int]bool
	response    Response
}

func (d *chatStreamDecoder) Decode(event sse.Event) ([]streamEvent, error) {
	if event.Data == "[DONE]" {
		if d.done {
			return nil, nil
		}
		d.done = true
		result := d.closeBlocks()
		return append(result, streamEvent{Type: streamDone, StopReason: defaultStopReason(d.response.StopReason), Usage: d.response.Usage}), nil
	}
	var value map[string]any
	if err := json.Unmarshal([]byte(event.Data), &value); err != nil {
		return nil, err
	}
	if id, ok := stringValue(value["id"]); ok {
		d.response.ID = id
	}
	if model, ok := stringValue(value["model"]); ok {
		d.response.Model = model
	}
	if usage := decodeChatUsage(value["usage"]); usage != nil {
		d.response.Usage = usage
	}
	var result []streamEvent
	if !d.started {
		d.started = true
		result = append(result, streamEvent{Type: streamStart, ID: d.response.ID, Model: d.response.Model})
	}
	choices, _ := array(value["choices"])
	for _, raw := range choices {
		choice, _ := object(raw)
		delta, _ := object(choice["delta"])
		if text, ok := stringValue(delta["content"]); ok && text != "" {
			if !d.textOpen {
				d.textIndex = d.nextBlock
				d.nextBlock++
				d.textOpen = true
				result = append(result, streamEvent{Type: streamContentStart, Index: d.textIndex, Content: Content{Type: ContentText}})
			}
			result = append(result, streamEvent{Type: streamContentDelta, Index: d.textIndex, Content: Content{Type: ContentText, Text: text}})
		}
		if reasoning, ok := stringValue(delta["reasoning_content"]); ok && reasoning != "" {
			index := d.nextBlock
			if existing, exists := d.toolIndexes[-1]; exists {
				index = existing
			} else {
				d.nextBlock++
				d.toolIndexes[-1] = index
				d.open[index] = true
				result = append(result, streamEvent{Type: streamContentStart, Index: index, Content: Content{Type: ContentReasoning}})
			}
			result = append(result, streamEvent{Type: streamContentDelta, Index: index, Content: Content{Type: ContentReasoning, Text: reasoning}})
		}
		if calls, ok := array(delta["tool_calls"]); ok {
			for _, rawCall := range calls {
				call, _ := object(rawCall)
				callIndex, _ := intValue(call["index"])
				index, exists := d.toolIndexes[callIndex]
				fn, _ := object(call["function"])
				if !exists {
					index = d.nextBlock
					d.nextBlock++
					d.toolIndexes[callIndex] = index
					d.open[index] = true
					id, _ := stringValue(call["id"])
					name, _ := stringValue(fn["name"])
					result = append(result, streamEvent{Type: streamContentStart, Index: index, Content: Content{Type: ContentToolCall, ID: id, Name: name, Arguments: map[string]any{}}})
				}
				if arguments, ok := stringValue(fn["arguments"]); ok && arguments != "" {
					result = append(result, streamEvent{Type: streamContentDelta, Index: index, Content: Content{Type: ContentToolCall}, ArgumentsDelta: arguments})
				}
			}
		}
		if finish, ok := stringValue(choice["finish_reason"]); ok && finish != "" {
			d.response.StopReason = chatStopReason(finish)
		}
	}
	return result, nil
}

func (d *chatStreamDecoder) Finish() []streamEvent {
	if d.done {
		return nil
	}
	result := d.closeBlocks()
	result = append(result, streamEvent{Type: streamDone, StopReason: defaultStopReason(d.response.StopReason), Usage: d.response.Usage})
	return result
}

func (d *chatStreamDecoder) closeBlocks() []streamEvent {
	var result []streamEvent
	if d.textOpen {
		result = append(result, streamEvent{Type: streamContentStop, Index: d.textIndex})
		d.textOpen = false
	}
	for _, index := range d.toolIndexes {
		if d.open[index] {
			result = append(result, streamEvent{Type: streamContentStop, Index: index})
			delete(d.open, index)
		}
	}
	return result
}

type streamEncoder interface {
	Encode(streamEvent) error
	Response() *Response
}

func newStreamEncoder(protocol Protocol, w http.ResponseWriter) (streamEncoder, error) {
	writer := newSSEWriter(w)
	base := streamEncoderBase{writer: writer, response: Response{}, blocks: map[int]*streamBlock{}}
	switch protocol {
	case Anthropic:
		return &anthropicStreamEncoder{streamEncoderBase: base}, nil
	case OpenAIResponses:
		return &responsesStreamEncoder{streamEncoderBase: base}, nil
	case OpenAIChat:
		return &chatStreamEncoder{streamEncoderBase: base}, nil
	default:
		return nil, fmt.Errorf("unsupported stream protocol %q", protocol)
	}
}

type streamBlock struct {
	content   Content
	arguments strings.Builder
}

type streamEncoderBase struct {
	writer   *sseWriter
	response Response
	blocks   map[int]*streamBlock
}

func (e *streamEncoderBase) apply(event streamEvent) {
	switch event.Type {
	case streamStart:
		e.response.ID = event.ID
		e.response.Model = event.Model
		if event.Usage != nil {
			e.response.Usage = event.Usage
		}
	case streamContentStart:
		e.blocks[event.Index] = &streamBlock{content: event.Content}
	case streamContentDelta:
		block := e.blocks[event.Index]
		if block == nil {
			block = &streamBlock{content: event.Content}
			e.blocks[event.Index] = block
		}
		if event.Content.Type == ContentText {
			block.content.Type = ContentText
			block.content.Text += event.Content.Text
		} else if event.Content.Type == ContentReasoning {
			block.content.Type = ContentReasoning
			block.content.Text += event.Content.Text
		} else {
			block.arguments.WriteString(event.ArgumentsDelta)
		}
	case streamContentStop:
		if block := e.blocks[event.Index]; block != nil {
			if block.content.Type == ContentToolCall {
				arguments, _ := parseArguments(block.arguments.String())
				block.content.Arguments = arguments
			}
			e.response.Content = append(e.response.Content, block.content)
		}
	case streamDone:
		e.response.StopReason = defaultStopReason(event.StopReason)
		if event.Usage != nil {
			e.response.Usage = event.Usage
		}
	}
}

func (e *streamEncoderBase) Response() *Response {
	return &e.response
}

type anthropicStreamEncoder struct{ streamEncoderBase }

func (e *anthropicStreamEncoder) Encode(event streamEvent) error {
	e.apply(event)
	switch event.Type {
	case streamStart:
		usage := map[string]any{"input_tokens": 0, "output_tokens": 0}
		if event.Usage != nil {
			usage["input_tokens"] = event.Usage.InputTokens
			usage["cache_read_input_tokens"] = event.Usage.CachedInputTokens
		}
		return e.writer.Send("message_start", map[string]any{"type": "message_start", "message": map[string]any{
			"id": ensureID(event.ID, "msg"), "type": "message", "role": "assistant", "model": event.Model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil, "usage": usage,
		}})
	case streamContentStart:
		var block map[string]any
		if event.Content.Type == ContentToolCall {
			block = map[string]any{"type": "tool_use", "id": event.Content.ID, "name": event.Content.Name, "input": map[string]any{}}
		} else if event.Content.Type == ContentReasoning {
			block = map[string]any{"type": "thinking", "thinking": "", "signature": ""}
		} else {
			block = map[string]any{"type": "text", "text": ""}
		}
		return e.writer.Send("content_block_start", map[string]any{"type": "content_block_start", "index": event.Index, "content_block": block})
	case streamContentDelta:
		var delta map[string]any
		if event.Content.Type == ContentToolCall {
			delta = map[string]any{"type": "input_json_delta", "partial_json": event.ArgumentsDelta}
		} else if event.Content.Type == ContentReasoning {
			delta = map[string]any{"type": "thinking_delta", "thinking": event.Content.Text}
		} else {
			delta = map[string]any{"type": "text_delta", "text": event.Content.Text}
		}
		return e.writer.Send("content_block_delta", map[string]any{"type": "content_block_delta", "index": event.Index, "delta": delta})
	case streamContentStop:
		return e.writer.Send("content_block_stop", map[string]any{"type": "content_block_stop", "index": event.Index})
	case streamDone:
		usage := map[string]any{"output_tokens": 0}
		if event.Usage != nil {
			usage["output_tokens"] = event.Usage.OutputTokens
		}
		if err := e.writer.Send("message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": encodeAnthropicStopReason(event.StopReason), "stop_sequence": nil}, "usage": usage}); err != nil {
			return err
		}
		return e.writer.Send("message_stop", map[string]any{"type": "message_stop"})
	}
	return nil
}

type responsesStreamEncoder struct {
	streamEncoderBase
	responseID string
}

func (e *responsesStreamEncoder) Encode(event streamEvent) error {
	e.apply(event)
	switch event.Type {
	case streamStart:
		e.responseID = ensureID(event.ID, "resp")
		skeleton := map[string]any{"id": e.responseID, "object": "response", "created_at": 0, "status": "in_progress", "model": event.Model, "output": []any{}}
		if err := e.writer.Send("response.created", map[string]any{"type": "response.created", "response": skeleton}); err != nil {
			return err
		}
		return e.writer.Send("response.in_progress", map[string]any{"type": "response.in_progress", "response": skeleton})
	case streamContentStart:
		if event.Content.Type == ContentToolCall {
			return e.writer.Send("response.output_item.added", map[string]any{"type": "response.output_item.added", "output_index": event.Index, "item": map[string]any{
				"type": "function_call", "id": fmt.Sprintf("fc_%d", event.Index), "status": "in_progress", "call_id": event.Content.ID, "name": event.Content.Name, "arguments": "",
			}})
		}
		if event.Content.Type == ContentReasoning {
			return e.writer.Send("response.output_item.added", map[string]any{"type": "response.output_item.added", "output_index": event.Index, "item": map[string]any{
				"type": "reasoning", "id": fmt.Sprintf("rs_%d", event.Index), "status": "in_progress", "summary": []any{},
			}})
		}
		itemID := fmt.Sprintf("msg_%d", event.Index)
		if err := e.writer.Send("response.output_item.added", map[string]any{"type": "response.output_item.added", "output_index": event.Index, "item": map[string]any{
			"type": "message", "id": itemID, "status": "in_progress", "role": "assistant", "content": []any{},
		}}); err != nil {
			return err
		}
		return e.writer.Send("response.content_part.added", map[string]any{"type": "response.content_part.added", "item_id": itemID, "output_index": event.Index, "content_index": 0, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}})
	case streamContentDelta:
		if event.Content.Type == ContentToolCall {
			return e.writer.Send("response.function_call_arguments.delta", map[string]any{"type": "response.function_call_arguments.delta", "item_id": fmt.Sprintf("fc_%d", event.Index), "output_index": event.Index, "delta": event.ArgumentsDelta})
		}
		if event.Content.Type == ContentReasoning {
			return e.writer.Send("response.reasoning_summary_text.delta", map[string]any{"type": "response.reasoning_summary_text.delta", "item_id": fmt.Sprintf("rs_%d", event.Index), "output_index": event.Index, "summary_index": 0, "delta": event.Content.Text})
		}
		return e.writer.Send("response.output_text.delta", map[string]any{"type": "response.output_text.delta", "item_id": fmt.Sprintf("msg_%d", event.Index), "output_index": event.Index, "content_index": 0, "delta": event.Content.Text})
	case streamContentStop:
		block := e.blocks[event.Index]
		if block == nil {
			return nil
		}
		if block.content.Type == ContentToolCall {
			arguments := block.arguments.String()
			if err := e.writer.Send("response.function_call_arguments.done", map[string]any{"type": "response.function_call_arguments.done", "item_id": fmt.Sprintf("fc_%d", event.Index), "output_index": event.Index, "arguments": arguments}); err != nil {
				return err
			}
			return e.writer.Send("response.output_item.done", map[string]any{"type": "response.output_item.done", "output_index": event.Index, "item": map[string]any{
				"type": "function_call", "id": fmt.Sprintf("fc_%d", event.Index), "status": "completed", "call_id": block.content.ID, "name": block.content.Name, "arguments": arguments,
			}})
		}
		if block.content.Type == ContentReasoning {
			if err := e.writer.Send("response.reasoning_summary_text.done", map[string]any{"type": "response.reasoning_summary_text.done", "item_id": fmt.Sprintf("rs_%d", event.Index), "output_index": event.Index, "summary_index": 0, "text": block.content.Text}); err != nil {
				return err
			}
			return e.writer.Send("response.output_item.done", map[string]any{"type": "response.output_item.done", "output_index": event.Index, "item": map[string]any{
				"type": "reasoning", "id": fmt.Sprintf("rs_%d", event.Index), "status": "completed", "summary": []any{map[string]any{"type": "summary_text", "text": block.content.Text}},
			}})
		}
		itemID := fmt.Sprintf("msg_%d", event.Index)
		if err := e.writer.Send("response.output_text.done", map[string]any{"type": "response.output_text.done", "item_id": itemID, "output_index": event.Index, "content_index": 0, "text": block.content.Text}); err != nil {
			return err
		}
		if err := e.writer.Send("response.content_part.done", map[string]any{"type": "response.content_part.done", "item_id": itemID, "output_index": event.Index, "content_index": 0, "part": map[string]any{"type": "output_text", "text": block.content.Text, "annotations": []any{}}}); err != nil {
			return err
		}
		return e.writer.Send("response.output_item.done", map[string]any{"type": "response.output_item.done", "output_index": event.Index, "item": map[string]any{
			"type": "message", "id": itemID, "status": "completed", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": block.content.Text, "annotations": []any{}}},
		}})
	case streamDone:
		body, _, err := EncodeResponse(OpenAIResponses, &e.response)
		if err != nil {
			return err
		}
		var response map[string]any
		if err := json.Unmarshal(body, &response); err != nil {
			return err
		}
		response["id"] = e.responseID
		return e.writer.Send("response.completed", map[string]any{"type": "response.completed", "response": response})
	}
	return nil
}

type chatStreamEncoder struct{ streamEncoderBase }

func (e *chatStreamEncoder) Encode(event streamEvent) error {
	e.apply(event)
	chunk := func(delta map[string]any, finish any, usage *Usage) map[string]any {
		value := map[string]any{"id": ensureID(e.response.ID, "chatcmpl"), "object": "chat.completion.chunk", "model": e.response.Model, "choices": []any{}}
		if delta != nil || finish != nil {
			value["choices"] = []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}
		}
		if usage != nil {
			value["usage"] = map[string]any{"prompt_tokens": usage.InputTokens, "completion_tokens": usage.OutputTokens, "total_tokens": usage.TotalTokens}
		}
		return value
	}
	switch event.Type {
	case streamStart:
		return e.writer.Data(chunk(map[string]any{"role": "assistant", "content": ""}, nil, nil))
	case streamContentStart:
		if event.Content.Type == ContentToolCall {
			return e.writer.Data(chunk(map[string]any{"tool_calls": []any{map[string]any{"index": event.Index, "id": event.Content.ID, "type": "function", "function": map[string]any{"name": event.Content.Name, "arguments": ""}}}}, nil, nil))
		}
	case streamContentDelta:
		if event.Content.Type == ContentToolCall {
			return e.writer.Data(chunk(map[string]any{"tool_calls": []any{map[string]any{"index": event.Index, "function": map[string]any{"arguments": event.ArgumentsDelta}}}}, nil, nil))
		}
		if event.Content.Type == ContentReasoning {
			return e.writer.Data(chunk(map[string]any{"reasoning_content": event.Content.Text}, nil, nil))
		}
		return e.writer.Data(chunk(map[string]any{"content": event.Content.Text}, nil, nil))
	case streamDone:
		if err := e.writer.Data(chunk(map[string]any{}, encodeChatStopReason(event.StopReason), nil)); err != nil {
			return err
		}
		if event.Usage != nil {
			if err := e.writer.Data(chunk(nil, nil, event.Usage)); err != nil {
				return err
			}
		}
		return e.writer.RawData("[DONE]")
	}
	return nil
}

type sseWriter struct {
	w       io.Writer
	flusher http.Flusher
}

func newSSEWriter(w http.ResponseWriter) *sseWriter {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	result := &sseWriter{w: w}
	if flusher, ok := w.(http.Flusher); ok {
		result.flusher = flusher
	}
	return result
}

func (w *sseWriter) Send(event string, value any) error {
	body, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w.w, "event: %s\ndata: %s\n\n", event, body); err != nil {
		return err
	}
	w.flush()
	return nil
}

func (w *sseWriter) Data(value any) error {
	body, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return w.RawData(string(body))
}

func (w *sseWriter) RawData(value string) error {
	if _, err := fmt.Fprintf(w.w, "data: %s\n\n", value); err != nil {
		return err
	}
	w.flush()
	return nil
}

func (w *sseWriter) flush() {
	if w.flusher != nil {
		w.flusher.Flush()
	}
}

func defaultStopReason(value string) string {
	if value == "" {
		return "stop"
	}
	return value
}
