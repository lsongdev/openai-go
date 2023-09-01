package openai

import (
	"encoding/json"
	"errors"
)

const (
	RoleSystem    = "system"
	RoleAssistant = "assistant"
	RoleUser      = "user"
)

const (
	GPT3_5_Trubo      = "gpt-3.5-turbo"
	GPT3_5_Trubo_0301 = "gpt-3.5-turbo-0301"
)

type ChatCompletionRequest struct {
	Model           string                  `json:"model,omitempty"`
	Messages        []ChatCompletionMessage `json:"messages,omitempty"`
	Temperature     int                     `json:"temperature,omitempty"`
	TopP            string                  `json:"top_p,omitempty"`
	NumberOfChoices int                     `json:"n,omitempty"`
	Stream          bool                    `json:"stream,omitempty"`
	Stop            []string                `json:"stop,omitempty"`
	MaxTokens       int                     `json:"max_tokens,omitempty"`
	User            string                  `json:"user,omitempty"`
}

type ChatCompletionResponse struct {
	Error   OpenAIErrorResponse    `json:"error,omitempty"`
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Choices []ChatCompletionChoice `json:"choices,omitempty"`
	Usage   CompletionUsage        `json:"usage,omitempty"`
}

// CompletionChoice represents one of possible completions.
type ChatCompletionChoice struct {
	Index        int                   `json:"index"`
	FinishReason string                `json:"finish_reason"`
	Message      ChatCompletionMessage `json:"message"`
}

type ChatCompletionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Create chat completion
// https://platform.openai.com/docs/api-reference/chat/create
func (openai *OpenAIClient) CreateChatCompletion(request ChatCompletionRequest) (resp ChatCompletionResponse, err error) {
	data, err := openai.MakeRequest("/chat/completions", request)
	if err != nil {
		return
	}
	json.NewDecoder(data).Decode(&resp)
	if resp.Error.Code != "" {
		err = errors.New(resp.Error.Message)
	}
	return
}
