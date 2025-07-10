import os
from dotenv import load_dotenv

def load_environment_variables():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
    return openai_api_key