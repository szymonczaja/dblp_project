from fastapi import FastAPI, HTTPException, Request
import os 
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import chromadb

PROMPT = ChatPromptTemplate.from_template("""
                You are a research assistant helping to find relevant academic papers.
                Based on the following papers from a database, answer the user's question.
                Always base your answer ONLY on the provided papers, not on your general knowledge.

                Relevant papers:
                {context}

                User question: {question}

                Provide a helpful answer referencing the specific papers above.
                """)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    print('Models downloading...')
    try:
        app.state.model = SentenceTransformer('all-MiniLM-L6-v2')
        app.state.client = chromadb.PersistentClient(path ='/app/data/chromadb')
        app.state.collection = app.state.client.get_collection('articles')
        app.state.llm = ChatGroq(model='llama-3.1-8b-instant', 
                                api_key=os.getenv('GROQ_API_KEY'), 
                                temperature=0.1)
        print('Models are ready!')
    except Exception as e:
        print(f'Loadig failed, error: {e}')
        raise
    yield 
    del app.state.model
    del app.state.client
    del app.state.collection 
    del app.state.llm
    print('\n Shutting down the service...')

app = FastAPI(title='DBLP ChatBot', version='1.0', lifespan=lifespan)

class QueryRequest(BaseModel):
    question : str
    n_results : int = 5
    class Config:
        extra = 'ignore'

@app.post('/query')
def query(body: QueryRequest, request: Request):
    model = request.app.state.model
    collection = request.app.state.collection 
    llm = request.app.state.llm 
    try:
        embedding = model.encode(body.question).tolist()
        results = collection.query(
            query_embeddings = [embedding],
            n_results = body.n_results,
            include = ['documents', 'metadatas'] 
        )
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        context = "\n\n".join([
                  f"Title: {doc}\nAuthors: {meta['authors']}\nYear: {meta['year']}\nCluster: {meta['cluster_name']}"
            for doc, meta in zip(docs, metas)
        ])
        chain = PROMPT | llm
        response = chain.invoke({
            'context' : context,
            'question' : body.question
        })
        return {
            'answer' : response.content,
            'sources' : results['metadatas'][0], 
            'titles' : results['documents'][0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health(request: Request):
    return {
        'status' : 'OK', 
        'model' : 'all-MiniLM-L6-v2', 
        'collection' : request.app.state.collection.name,
        'llm' : 'llama-3.1-8b-instant'
    }
