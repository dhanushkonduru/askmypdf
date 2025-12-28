import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY not found! Please set it in your .env file or as an environment variable.\n"
            "Get your API key from: https://console.groq.com/keys"
        )
    return Groq(api_key=key)
