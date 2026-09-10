package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGetFirstChoiceReturnsSliceElement(t *testing.T) {
	resp := ChatCompletionResponse{Choices: []ChatCompletionChoice{{FinishReason: "stop"}}}
	choice := resp.GetFirstChoice()
	if choice == nil {
		t.Fatal("missing first choice")
	}
	choice.FinishReason = "length"
	if resp.Choices[0].FinishReason != "length" {
		t.Fatal("GetFirstChoice returned a copy")
	}
}

func TestCreateChatCompletionStreamForcesStreamWithoutMutation(t *testing.T) {
	seenStream := make(chan bool, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		seenStream <- req.Stream
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	client, _ := NewClient(&Configuration{API: server.URL})
	request := &ChatCompletionRequest{Model: "gpt-test"}
	stream, err := client.CreateChatCompletionStream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
	}
	if !<-seenStream {
		t.Fatal("stream request did not set stream=true")
	}
	if request.Stream {
		t.Fatal("CreateChatCompletionStream mutated its request")
	}
}
