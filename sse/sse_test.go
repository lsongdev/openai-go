package sse

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestReadEventsHandlesCRLFMultilineDataAndEOF(t *testing.T) {
	source := strings.NewReader(": heartbeat\r\nid: 7\r\nevent: message\r\ndata: {\"text\":\r\ndata: \"hello\"}\r\n\r\ndata: [DONE]")
	var events []Event
	err := Read(context.Background(), source, func(event Event) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 {
		t.Fatalf("events = %#v, want 2", events)
	}
	if events[0].Type != "message" || events[0].ID != "7" || events[0].Data != "{\"text\":\n\"hello\"}" {
		t.Errorf("first event = %#v", events[0])
	}
	if events[1].Data != "[DONE]" {
		t.Errorf("second event = %#v", events[1])
	}
}

func TestReadEventsPreservesDataWhitespace(t *testing.T) {
	var got Event
	err := Read(context.Background(), strings.NewReader("data:  leading and trailing  \n\n"), func(event Event) error {
		got = event
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Data != " leading and trailing  " {
		t.Fatalf("data = %q", got.Data)
	}
}

func TestDoIncludesUpstreamErrorBody(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadRequest,
			Status:     "400 Bad Request",
			Body:       io.NopCloser(strings.NewReader(`{"error":"bad model"}`)),
			Header:     make(http.Header),
		}, nil
	})}
	req, err := http.NewRequest(http.MethodGet, "https://example.test/stream", nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Do(context.Background(), client, req)
	if err == nil || !strings.Contains(err.Error(), "bad model") {
		t.Fatalf("error = %v", err)
	}
}

func TestReadEventsPropagatesEmitterError(t *testing.T) {
	want := errors.New("stop")
	err := Read(context.Background(), strings.NewReader("data: one\n\n"), func(Event) error { return want })
	if !errors.Is(err, want) {
		t.Fatalf("error = %v, want %v", err, want)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}
