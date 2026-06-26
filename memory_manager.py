from mem0 import Memory
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONNECTION=os.getenv("CONNECTION")
user=os.getenv("USER")

config={
    "vector_store":{
    "provider":"pgvector",
    "config": {
        "connection_string": CONNECTION
        }
    },
    "llm":{
        "provider":"openai",
        "config":{
            "model":"gpt-4.1-mini",
            "api_key":OPENAI_API_KEY
        }

    },
    "embedder":{
        "provider":"openai",
        "config":{
            "model":"text-embedding-3-small",
            "api_key":OPENAI_API_KEY
        }
    }
}

memory=Memory.from_config(config)

def retrieve_memory(query):
    """ Searches for relevant data from stored long term memory"""
    mem_data = memory.search(
        query=query,
        filters={
            "user_id": user
        }
    )

    content = [i["memory"] for i in mem_data["results"]]
    memory_text = "\n".join(content)
    return memory_text

def store_memory(query,data):
    """Stores the responses"""

    memory.add(
        messages=[
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": data
            }
        ],
        user_id=user
    )
