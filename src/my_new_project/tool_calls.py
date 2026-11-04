import json

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# --------------------------------------------------------------
# 0. Connect to Ollama using OpenAI-compatible API
# --------------------------------------------------------------

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

model = "llama3.2"


# --------------------------------------------------------------
# 1. Define the actual Python function
# --------------------------------------------------------------


def get_weather(latitude: float, longitude: float):
    """Get current weather for the given coordinates."""

    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&current=temperature_2m,wind_speed_10m"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )

    response.raise_for_status()

    data = response.json()

    return data["current"]


# --------------------------------------------------------------
# 2. Define the tool schema
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature and wind speed for provided coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the location.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the location.",
                    },
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
        },
    }
]


# --------------------------------------------------------------
# 3. Prepare conversation
# --------------------------------------------------------------

system_prompt = """
You are a helpful weather assistant.

When the user asks about weather in a city:
1. Identify the city's latitude and longitude.
2. Call the get_weather function with those coordinates.
3. Use the returned data to answer the user.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Paris today?"},
]


# --------------------------------------------------------------
# 4. Call model with tool definitions
# --------------------------------------------------------------

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
)


# --------------------------------------------------------------
# 5. Extract the function call
# --------------------------------------------------------------

assistant_message = completion.choices[0].message

if assistant_message.tool_calls:
    # Add assistant's tool-call message to conversation
    messages.append(assistant_message.model_dump(exclude_none=True))

    # Execute each requested tool
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name

        function_args = json.loads(tool_call.function.arguments)

        print(f"Tool called: {function_name}")
        print(f"Arguments: {function_args}")

        # ------------------------------------------------------
        # 6. Execute the Python function
        # ------------------------------------------------------

        if function_name == "get_weather":
            function_result = get_weather(**function_args)

            print("Raw result:", function_result)

            # --------------------------------------------------
            # 7. Send function result back to the model
            # --------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_result),
                }
            )

    # ----------------------------------------------------------
    # 8. Get final response from model
    # ----------------------------------------------------------

    final_completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    print("\nFinal answer:")
    print(final_completion.choices[0].message.content)

else:
    print(assistant_message.content)


# --------------------------------------------------------------
# 9. Optional: Validate final response with Pydantic
# --------------------------------------------------------------


class WeatherResponse(BaseModel):
    temperature: float = Field(description="Current temperature in Celsius.")

    response: str = Field(
        description="Natural language response to the user's question."
    )


structured_completion = client.chat.completions.parse(
    model=model,
    messages=messages
    + [
        {
            "role": "user",
            "content": """
            Return the current temperature and a natural language
            weather explanation using the required JSON structure.
            """,
        }
    ],
    response_format=WeatherResponse,
)

raw_response = structured_completion.choices[0].message.content

print("Raw response:", raw_response)

final_response = WeatherResponse.model_validate_json(raw_response)

print(final_response)
