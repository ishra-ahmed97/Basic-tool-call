from openai import OpenAI
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    city: str


client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

result = client.beta.chat.completions.parse(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": """
        You are a helpful assistant.
        Generate realistic fictional people.
        The name must be a valid human full name.
        Do not use symbols, brackets, or random characters.
        """,
        },
        {
            "role": "user",
            "content": "Give me the name, age and city of a fictional person.",
        },
    ],
    response_format=Person,
)

person = result.choices[0].message.parsed

print(person)
print(person.name)
print(person.age)
print(person.city)
