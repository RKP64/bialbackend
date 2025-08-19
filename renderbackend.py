import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import openai
from openai import AzureOpenAI
import json
import base64
import os
import pandas as pd
import tempfile
import html
import traceback
import re
import docx
import requests
from docx import Document
import io
from bs4 import BeautifulSoup
from fastapi.responses import FileResponse
import hashlib
from datetime import datetime
from pymongo import MongoClient
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Pydantic Models for Configuration ---
class AzureCredentials(BaseModel):
    cosmos_mongo_connection_string: str = os.environ.get("COSMOS_MONGO_CONNECTION_STRING")
    cosmos_database_name: str = os.environ.get("COSMOS_DATABASE_NAME")
    cosmos_logs_collection: str = os.environ.get("COSMOS_LOGS_COLLECTION")
    cosmos_feedback_collection: str = os.environ.get("COSMOS_FEEDBACK_COLLECTION")
    content_safety_endpoint: str = os.environ.get("CONTENT_SAFETY_ENDPOINT")
    content_safety_key: str = os.environ.get("CONTENT_SAFETY_KEY")
    app_insights_connection_string: str = os.environ.get("APP_INSIGHTS_CONNECTION_STRING")
    search_endpoint: str = os.environ.get("SEARCH_ENDPOINT")
    search_api_key: str = os.environ.get("SEARCH_API_KEY")
    default_search_index_name: str = os.environ.get("DEFAULT_SEARCH_INDEX_NAME")
    default_vector_field_name: str = os.environ.get("DEFAULT_VECTOR_FIELD_NAME")
    default_semantic_config_name: str = os.environ.get("DEFAULT_SEMANTIC_CONFIG_NAME")
    openai_endpoint: str = os.environ.get("OPENAI_ENDPOINT")
    openai_api_version: str = os.environ.get("OPENAI_API_VERSION")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY")
    deployment_id: str = os.environ.get("DEPLOYMENT_ID")
    planning_llm_deployment_id: str = os.environ.get("PLANNING_LLM_DEPLOYMENT_ID")
    embedding_deployment_id: str = os.environ.get("EMBEDDING_DEPLOYMENT_ID")

# --- Global Configuration and Clients ---
config = AzureCredentials()
app = FastAPI(title="BIAL Regulatory Platform API")

# --- CORS Middleware ---
# **FIX:** Explicitly allowing all methods and headers. This is crucial for
# handling the browser's preflight (OPTIONS) requests correctly, especially
# when deployed behind a proxy like Render's.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Service Clients (initialized on startup) ---
class ServiceClients:
    def __init__(self):
        self.synthesis_openai_client = AzureOpenAI(api_key=config.openai_api_key, azure_endpoint=config.openai_endpoint, api_version=config.openai_api_version)
        self.planning_openai_client = self.synthesis_openai_client
        self.search_query_embeddings_model = AzureOpenAIEmbeddings(azure_deployment=config.embedding_deployment_id, azure_endpoint=config.openai_endpoint, api_key=config.openai_api_key, api_version=config.openai_api_version, chunk_size=1)
        self.mongo_client = MongoClient(config.cosmos_mongo_connection_string) if not self.check_creds(config.cosmos_mongo_connection_string) else None

    def check_creds(self, cred_value):
        return not cred_value or "YOUR_" in cred_value.upper()

clients = ServiceClients()

# --- Pydantic Models for API Requests & Responses ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []
    user: str = "anonymous"

class ChatResponse(BaseModel):
    answer: str

# ... (The rest of your code remains the same) ...

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str
    user: str = "anonymous"

class AnalysisResponse(BaseModel):
    report: str

class RefineReportRequest(BaseModel):
    original_report: str
    new_info: str

class DownloadRequest(BaseModel):
    html_content: str

# --- Core Logic Functions ---
def extract_text_from_docx(file: UploadFile) -> str:
    try:
        file_content = file.file.read()
        file_stream = io.BytesIO(file_content)
        document = docx.Document(file_stream)
        text = "\n".join([para.text for para in document.paragraphs])
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading Word document: {e}")

def get_query_vector(text_to_embed: str):
    try:
        return clients.search_query_embeddings_model.embed_query(text_to_embed)
    except Exception as e:
        print(f"Error generating query vector: {e}")
        return None

def query_azure_search(query_text: str, index_name: str, k: int = 5):
    context = ""
    try:
        search_client = SearchClient(config.search_endpoint, index_name, AzureKeyCredential(config.search_api_key))
        search_kwargs = {"search_text": query_text if query_text and query_text.strip() else "*", "top": k}
        if (query_vector := get_query_vector(query_text)):
            search_kwargs["vector_queries"] = [VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields=config.default_vector_field_name)]
            search_kwargs.update({"query_type": "semantic", "semantic_configuration_name": config.default_semantic_config_name})
        
        results = search_client.search(**search_kwargs)
        context = "\n\n".join([doc.get("content", "") for doc in results])
        return context.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing search index: {e}")

def get_query_plan_from_llm(user_question: str, history: List[ChatMessage]):
    history_formatted = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
    planning_prompt = f"""You are a query planning assistant...""" # Truncated for brevity
    try:
        response = clients.planning_openai_client.chat.completions.create(model=config.planning_llm_deployment_id, messages=[{"role": "user", "content": planning_prompt}], temperature=0.0, max_tokens=1000)
        plan_str = response.choices[0].message.content
        if match := re.search(r'\[.*\]', plan_str, re.DOTALL):
            return json.loads(match.group(0))
        return [user_question]
    except Exception as e:
        print(f"Error in query planner: {e}")
        return [user_question]

def generate_answer_from_search(user_question: str, history: List[ChatMessage], system_prompt: str):
    query_plan = get_query_plan_from_llm(user_question, history)
    combined_context = "\n\n".join([query_azure_search(q, config.default_search_index_name) for q in query_plan])
    
    if not combined_context.strip():
        return "No relevant information found in the knowledge base."

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": f"Based on the conversation and new context, answer: {user_question}\n\nCONTEXT:\n{combined_context}"})
    
    try:
        response = clients.synthesis_openai_client.chat.completions.create(model=config.deployment_id, messages=messages, temperature=0.5, max_tokens=4000)
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

# --- Analysis Prompts Config ---
analysis_prompts_config = {
    "MDA Manpower Analysis": {
        "Analysis of manpower expenditure projection for BIAL for fourth control period": "...",
        "Analysis of actual manpower expenditure for BIAL for third control period": "..."
    },
    "Utility Analysis": {
        "Analysis of electricity expenditure projection for BIAL for third control period": "...",
        "Analysis of water expenditure projection for BIAL for third control period": "..."
    },
    "R&M Analysis": {
        "Projected R&M Expenditure Analysis": "...",
        "Actual R&M Expenditure True-Up Analysis": "..."
    }
}

# --- API Endpoints ---
@app.post("/generate-initial-report")
async def generate_initial_report(analysis_title: str, file: UploadFile = File(...)):
    # ... endpoint logic ...
    pass

@app.post("/chat", response_model=ChatResponse)
async def conversational_chat(request: ChatRequest):
    system_prompt = """You are an AI assistant for Multiyear Tariff submission for AERA...""" # Truncated for brevity
    answer = generate_answer_from_search(user_question=request.question, history=request.history, system_prompt=system_prompt)
    return ChatResponse(answer=answer)

@app.post("/mda-chat")
async def mda_chat(request: ChatRequest):
    system_prompt = "You are an expert AI assistant for AERA regulatory matters..."
    answer = generate_answer_from_search(user_question=request.question, history=request.history, system_prompt=system_prompt)
    return {"answer": answer}

@app.post("/refine-report")
async def refine_report(request: RefineReportRequest):
    # ... endpoint logic ...
    pass

@app.post("/feedback")
async def handle_feedback(request: FeedbackRequest):
    # ... endpoint logic ...
    pass

@app.post("/download-report")
async def download_report(request: DownloadRequest):
    # ... endpoint logic ...
    pass

@app.get("/")
def read_root():
    return {"message": "BIAL Regulatory Platform API is running."}

