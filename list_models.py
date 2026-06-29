import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in .env")
    exit(1)

headers = {
    "Authorization": f"Bearer {api_key}"
}
url = "https://api.groq.com/openai/v1/models"

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    models = response.json().get("data", [])
    print("Available Groq models on your API Key:")
    for model in models:
        print(f"- {model['id']}")
except Exception as e:
    print(f"Error fetching models: {e}")
