package chatgpt

type ChatGPT struct {
}

func NewClient() (chatgpt *ChatGPT) {
	chatgpt = &ChatGPT{}
	return
}

func (chatgpt *ChatGPT) GetAccessToken() (accessToken string) {
	return
}

func (chatgpt *ChatGPT) SendMessage() (output string) {
	return
}
