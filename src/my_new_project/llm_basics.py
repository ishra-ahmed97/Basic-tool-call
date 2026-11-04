from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
model = "llama3.2"

result = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is the square root of 49?"}],
)

print(result)

# ---- 4. Multi-turn conversation (passing a list of messages) ----
result = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Ishra"},
        {"role": "assistant", "content": "Nice to meet you Ishra!"},
        {"role": "user", "content": "What is my name?"},
    ],
)

print(result.choices[0].message.content)


result = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant who makes jokes."},
        {"role": "user", "content": "What is the square root of 49?"},
    ],
    temperature=0.7,
    max_tokens=100,
)

print(result.choices[0].message.content)
print("Prompt tokens:", result.usage.prompt_tokens)
print("Completion tokens:", result.usage.completion_tokens)
print("Total tokens:", result.usage.total_tokens)


stream = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short story about a robot."},
    ],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content

    if content:
        print(content, end="", flush=True)
