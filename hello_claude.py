from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=64,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)

print(message.content[0].text)
