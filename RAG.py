from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()
import os

key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4.1-mini",api_key=key)
worker=ChatOpenAI(model="gpt-4.1-mini",api_key=key)

pdf = ["sample1.pdf", "sample2.pdf"]
doc = []
for i in pdf:
    loader = PyPDFLoader(i)
    doc.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(doc)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstores = FAISS.from_documents(docs, embedding)

retriever = vectorstores.as_retriever(search_type="mmr", search_kwargs={"k": 5}) #retrieving top 5 chunks

def is_required_rag(query): #decides if rag is required, else simple LLM answer
    prompt = f"""
        You are a decision agent.

        Decide if this query requires external knowledge retrieval.

        Return ONLY:
        YES or NO

        Query:
        {query}
        """
    response=llm.invoke(prompt).content.strip().upper()
    if "YES" in response:
        return True
    else:
        return False

def is_relevant(query,content):
    prompt = f"""
        You are a strict evaluator.

        Determine if the context is sufficient to answer the query.

        Query:
        {query}

        Context:
        {content}

        Return ONLY:
        YES or NO
        """
    response=llm.invoke(prompt).content.strip().upper()
    if "YES" in response:
        return True
    else:
        return False

def re_write_query(query,history):
    history_text = ""
    for h in history[-3:]:
        history_text += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
    prompt = f"""
    You are a query rewriter.

    Your task is to convert the user's query into a clear, complete, standalone query.

    Instructions:
    1. Resolve references like "it", "this", "that", "they" using the conversation history
    2. Add missing context so the query is fully understandable on its own
    3. Keep the original intent EXACTLY the same
    4. Do NOT add new information or assumptions
    5. Do NOT answer the query
    6. Return ONLY the rewritten query

    Conversation History:
    {history_text}

    User Query:
    {query}
    """
    response=worker.invoke(prompt).content
    return response

def score_chunk(query,content):
    prompt = f"""
        You are a relevance scorer.

        Score how relevant the following context is to the query.

        Query:
        {query}

        Context:
        {content}

        STRICTLY Return ONLY a number between 1 and 10.
        """
    try:
        response = worker.invoke(prompt).content
        return int(response)
    except:
        return 0

def re_ranking_agent(query,docs):
    scored_docs=[]
    for d in docs:
        score=score_chunk(query,d.page_content)
        scored_docs.append((d,score))

    scored_docs=sorted(scored_docs,key=lambda x: x[1],reverse=True)
    final_docs=[i for i,j in scored_docs[:2]] #keeping only top 2 scored chunks
    return final_docs

def run_RAG(query,history):

    new_query=re_write_query(query,history)
    history_text=""
    for i in history[-3:]:
        history_text += f"User: {i['user']}\nAssistant: {i['assistant']}\n"

    augmented_query= new_query +"\nContext: \n"+history_text #using history to influence chunk retrieval
    context = retriever.invoke(augmented_query)
    docs=re_ranking_agent(new_query,context) #used re-written query
    content = ""
    source = []
    for d in docs:
        content += d.page_content + "\n"
        source.append(f"{d.metadata.get('source')} (Page: {d.metadata.get('page')})")
    return content, "\n".join(set(source))
