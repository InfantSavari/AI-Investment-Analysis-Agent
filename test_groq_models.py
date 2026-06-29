import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

models_to_test = [
    "qwen/qwen3.6-27b",
    "allam-2-7b",
    "groq/compound-mini",
    "openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b-instant"
]

for model in models_to_test:
    print(f"Testing model: {model}...")
    try:
        llm = ChatGroq(model=model, temperature=0.2, max_retries=1)
        response = llm.invoke("Hello, say 'OK'")
        print(f"  [SUCCESS] {model}: {response.content.strip()}")
    except Exception as e:
        print(f"  [FAILED] {model}: {e}")
