import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

models_to_test = [
    "openai/gpt-oss-120b",
    "groq/compound",
    "canopylabs/orpheus-v1-english"
]

for model in models_to_test:
    print(f"Testing model: {model}...")
    try:
        llm = ChatGroq(model=model, temperature=0.2, max_retries=1)
        response = llm.invoke("Hello, say 'OK'")
        print(f"  [SUCCESS] {model}: {response.content.strip()}")
    except Exception as e:
        print(f"  [FAILED] {model}: {e}")
