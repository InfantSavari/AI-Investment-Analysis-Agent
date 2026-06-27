import httpx
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"

try:
    r = httpx.get(url)
    r.raise_for_status()
    models = r.json().get("models", [])
    valid_models = [m["name"].replace("models/", "") for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
    
    print("\n--- AVAILABLE GEMINI MODELS ---")
    for m in valid_models:
        print(f"- {m}")
except Exception as e:
    print(f"Error fetching models: {e}")
