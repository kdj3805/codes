# chat_example.py

import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain what Groq is in two sentences.",
        }
    ],
    model="llama-3.1-8b-instant",# fast, free-tier model
    temperature=0.3,   
)
print(chat_completion.choices[0].message.content)
