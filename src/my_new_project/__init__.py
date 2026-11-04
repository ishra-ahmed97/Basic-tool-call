import os

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Now use your variables
api_key = os.environ.get("OPENAI_API_KEY")

print(f"API Key: {api_key}")
