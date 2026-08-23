package sse

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type Event struct {
	Type  string
	Data  string
	ID    string
	Retry string
}

type Stream struct {
	Events chan Event
	err    chan error
}

func NewStream(events chan Event, err chan error) *Stream {
	return &Stream{Events: events, err: err}
}

func (s *Stream) Close() {
	close(s.Events)
	close(s.err)
}

func (s *Stream) Err() <-chan error {
	return s.err
}

func Do(ctx context.Context, client *http.Client, req *http.Request) (*Stream, error) {
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sse request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("sse unexpected status %s: %s", resp.Status, string(body))
	}

	stream := NewStream(make(chan Event), make(chan error, 1))

	go func() {
		defer resp.Body.Close()
		defer stream.Close()

		err := Read(ctx, resp.Body, func(event Event) error {
			select {
			case stream.Events <- event:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		if err != nil && ctx.Err() == nil {
			stream.err <- err
		}
	}()

	return stream, nil
}

// Read decodes an SSE stream and calls emit for each complete event.
func Read(ctx context.Context, source io.Reader, emit func(Event) error) error {
	reader := bufio.NewReader(source)
	var event Event
	var data []string
	dispatch := func() error {
		if len(data) == 0 {
			event = Event{}
			return nil
		}
		event.Data = strings.Join(data, "\n")
		if err := emit(event); err != nil {
			return err
		}
		event = Event{}
		data = nil
		return nil
	}

	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimSuffix(line, "\n")
			line = strings.TrimSuffix(line, "\r")
			if line == "" {
				if dispatchErr := dispatch(); dispatchErr != nil {
					return dispatchErr
				}
			} else if !strings.HasPrefix(line, ":") {
				field, value, _ := strings.Cut(line, ":")
				value = strings.TrimPrefix(value, " ")
				switch field {
				case "event":
					event.Type = value
				case "data":
					data = append(data, value)
				case "id":
					if !strings.ContainsRune(value, '\x00') {
						event.ID = value
					}
				case "retry":
					event.Retry = value
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				return dispatch()
			}
			return err
		}
	}
}
