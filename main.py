from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# -- Import CortexFlow agent --

from graph import agent
from memory_manager import retrieve_memory,store_memory

app=FastAPI()
history=[]

class ChatRequest(BaseModel):
    query:str

class ChatResponse(BaseModel):
    response:str

@app.get("/")
def home():
    return {"message":"Welcome to CortexFlow!"}
@app.post("/chat",response_model=ChatResponse)
def chat(request: ChatRequest):
    global history
    try:
        query=request.query
        try:
            memory_text = retrieve_memory(query)
        except Exception as e:
            memory_text = ""

        ans = agent.invoke({"query": query, "history": history, "memory": memory_text})

        history = ans.get("history", history)

        store_memory(query, ans["final"])

        return {"response":ans["final"]}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )




