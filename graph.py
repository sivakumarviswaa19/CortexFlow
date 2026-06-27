# --- IMPORTS ---

from RAG import run_RAG
from langchain_openai import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing import List, TypedDict, Dict
import asyncio

# --Memory integration--
from memory_manager import *

from dotenv import load_dotenv

load_dotenv()

from mcp_use import MCPAgent
from mcp_client import client

import os
import re
import numexpr
import json

# --- ENV SETUP ---
key = os.getenv("OPENAI_API_KEY")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

llm = ChatOpenAI(model="gpt-4.1-mini", api_key=key)
worker = ChatOpenAI(model="gpt-4.1-mini", api_key=key)

# mcp agent
agent = MCPAgent(client=client, llm=worker)

search = DuckDuckGoSearchRun()


# --- MATH AGENT ---
def math_agent(query):
    try:
        match = re.search(r'[\d\.\+\-\*\/\(\)]+', query)
        if not match:
            return "No valid math expression found"
        return str(numexpr.evaluate(match.group()))
    except:
        return "Invalid mathematical expression"


# --- CODING AGENT ---
def coding_agent(query):
    prompt = f"""
    You are a coding assistant.
    Answer clearly with correct code if needed.

    Query:
    {query}
    """
    return worker.invoke(prompt).content


# --- NOTES AGENT ---
def filesystem_agent(query):
    read_flag = 0
    if any(word in query.lower() for word in [
        "read",
        "show",
        "retrieve",
        "what did i save",
        "display"
    ]):
        read_flag = 1

    path = os.getenv("path")

    async def sub():
        await client.create_all_sessions()
        session = client.get_session("filesystem")
        result = await session.call_tool(
            "read_file",
            {
                "path": path,
            }
        )
        return result.content[0].text

    if read_flag:
        return asyncio.run(sub())
    else:
        async def sub_main():
            content = worker.invoke(f"""Understand this and only return the test or main message of the query,
                                    Eg. if query is 'Save understood mcp in notes' then return 'understood mcp'
                                    Given query: {query}""").content
            await client.create_all_sessions()
            data = await sub()
            session = client.get_session("filesystem")
            result = await session.call_tool(
                "write_file",
                {
                    "path": path,
                    "content": data + "\n" + content
                }
            )
            return result.content[0].text

        return asyncio.run(sub_main())


# --- SEARCH AGENT (RAW) ---
def search_agent(query):
    return search.run(query)


# --- GENERATOR (CORE BRAIN) ---
def generator_agent(query, data, history, memory):
    history_text = ""
    for h in history[-3:]:
        history_text += f"User: {h['user']}\nAssistant: {h['assistant']}\n"

    prompt = f"""
    You are a decision-making assistant.

    RULES:
    - Give a clear answer
    - Do NOT say "it depends"
    - Be decisive

    Long-term memories about the user:
    {memory}

    Use these memories only when they are relevant to the user's question.
    Ignore unrelated memories.

    Conversation History:
    {history_text}

    Context:
    {data}

    Query:
    {query}

    Output:

    Answer:

    - Return the final answer in plain text ONLY

    - Do NOT use LaTeX

    - Do NOT use \\boxed{{}}

    - Just give the answer normally (e.g., 8, not \\boxed{{8}})

    Reasoning:
    - point 1
    - point 2
    """

    return worker.invoke(prompt).content


# --- SIMPLIFIER ---
def simplify_agent(content, query):
    prompt = f"""
    Simplify this answer:
    - short
    - clear
    - beginner-friendly

    Content:
    {content}
    """
    return worker.invoke(prompt).content


# --- CRITIC ---
def critic_agent(content, query):
    prompt = f"""
    You are a strict critic.

    Return ONLY:
    - GOOD
    - BAD: <reason>

    Content:
    {content}
    """
    return llm.invoke(prompt).content


# --- IMPROVE ---
def improve_agent(content, feedback):
    prompt = f"""
    Improve this answer based on feedback.

    Content:
    {content}

    Feedback:
    {feedback}
    """
    return worker.invoke(prompt).content


# --- STATE ---
class State(TypedDict):
    query: str
    memory: str
    type: str
    complexity: str
    final: str
    simplifier: str
    critic: str
    history: List[Dict[str, str]]


# --- PLAN + DECIDE ---
def plan_decide_agent(state):
    query = state["query"]

    prompt = f"""
    You are the planner for CortexFlow.

    Your job is to decide:
    1. The best tool to use.
    2. Whether the task is simple or complex.

    IMPORTANT:
    Long-term memory has ALREADY been retrieved before you are called.
    If the retrieved memory is sufficient to answer the user's question,
    DO NOT choose another tool just because the question refers to past conversations,
    preferences, names, or remembered information.

    Available Tools:

    - math
        Arithmetic, formulas, statistics, equations, unit conversions.

    - coding
        Programming, debugging, code generation, software engineering.

    - search
        Internet knowledge, current events, recent information,
        information unlikely to exist in memory or the local knowledge base.

    - rag
        Questions about uploaded PDFs, local documents,
        or the project's vector database.

    - notes
        ONLY when the user explicitly wants to interact with local files.

        Examples:
        - Save this to notes
        - Create a file
        - Write this into notes.txt
        - Read notes.txt
        - Update this file
        - Delete this note
        - List my files

        NEVER choose notes simply because the user asks:
        - "What do you remember?"
        - "What is my name?"
        - "What did I tell you?"
        - "What laptop do I use?"
        - "What was my last query?"
        Those should be answered using the already retrieved long-term memory.

    - reasoning
        General reasoning, explanations, brainstorming,
        planning, comparisons, conversations,
        or when no specialised tool is needed.

    Complexity:

    - simple
    - complex

    Return ONLY valid JSON:

    {{
        "type": "...",
        "complexity": "..."
    }}

    Query:
    {query}
    """

    response = llm.invoke(prompt).content.strip()

    try:
        if "```" in response:
            response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(response)
        return {"type": data["type"], "complexity": data["complexity"]}
    except:
        return {"type": "reasoning", "complexity": "simple"}


# --- EXECUTOR ---
def executor_agent(state):
    query = state["query"]
    history = state.get("history", [])
    decision = state["type"]

    if decision == "math":
        return {"final": math_agent(query)}

    elif decision == "notes":
        return {"final": filesystem_agent(query)}

    elif decision == "coding":
        return {"final": coding_agent(query)}

    elif decision == "search":
        return {"final": search_agent(query)}

    elif decision == "rag":
        content, sources = run_RAG(query, history)
        return {"final": content}

    else:
        return {"final": worker.invoke(query).content}


# --- GENERATOR NODE ---
def generator_node(state):
    query = state["query"]
    history = state.get("history", [])
    data = state["final"]
    memory = state.get("memory", "")

    result = generator_agent(query, data, history, memory)

    return {"final": result}


# --- SIMPLIFIER NODE ---
def simplifier_node(state):
    content = state["final"]
    query = state["query"]

    return {"simplifier": simplify_agent(content, query)}


# --- CRITIC NODE ---
def critic_node(state):
    return {"critic": critic_agent(state["simplifier"], state["query"])}


# --- IMPROVE NODE ---
def improve_node(state):
    return {
        "simplifier": improve_agent(state["simplifier"], state["critic"])
    }


# --- ROUTING ---
def route_after_executor(state):
    query = state["query"]
    tool = state["type"]
    if tool == "math":
        return "finalize"
    elif tool == "notes":
        return "finalize"
    return "generator"


def route_after_generator(state):
    if state["complexity"] == "simple":
        return "finalize"
    return "simplifier_node"


def route_critic(state):
    if "GOOD" in state["critic"]:
        return "finalize"
    return "improve_node"


# --- FINALIZE ---
def finalize_node(state):
    query = state["query"]
    history = state.get("history", [])

    final_ans = state.get("simplifier", state["final"])

    history.append({"user": query, "assistant": final_ans})
    history = history[-3:]

    return {"final": final_ans, "history": history}


# --- GRAPH ---

graph = StateGraph(State)

graph.add_node("plan_decide", plan_decide_agent)
graph.add_node("executor", executor_agent)
graph.add_node("generator", generator_node)
graph.add_node("simplifier_node", simplifier_node)
graph.add_node("critic_node", critic_node)
graph.add_node("improve_node", improve_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("plan_decide")

graph.add_edge("plan_decide", "executor")
graph.add_conditional_edges("executor", route_after_executor)

graph.add_conditional_edges("generator", route_after_generator)

graph.add_edge("simplifier_node", "critic_node")
graph.add_conditional_edges("critic_node", route_critic)

graph.add_edge("improve_node", "critic_node")
graph.add_edge("finalize", END)

agent = graph.compile()

