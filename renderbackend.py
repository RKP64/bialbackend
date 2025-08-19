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
# This line should be at the top to load variables from .env file
load_dotenv()

# --- Pydantic Models for Configuration ---
# All values are now loaded from environment variables
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Service Clients (initialized on startup) ---
class ServiceClients:
    def __init__(self):
        self.synthesis_openai_client = AzureOpenAI(api_key=config.openai_api_key, azure_endpoint=config.openai_endpoint, api_version=config.openai_api_version)
        self.planning_openai_client = self.synthesis_openai_client
        self.search_query_embeddings_model = AzureOpenAIEmbeddings(azure_deployment=config.embedding_deployment_id, azure_endpoint=config.openai_endpoint, api_key=config.openai_api_key, api_version=config.openai_api_version, chunk_size=1)
        self.mongo_client = MongoClient(config.cosmos_mongo_connection_string) if not self.check_creds(config.cosmos_mongo_connection_string) else None

    def check_creds(self, cred_value):
        # This check is useful for optional credentials
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
    planning_prompt = f"""You are a query planning assistant specializing in breaking down complex questions about **AERA regulatory documents, often concerning tariff orders, consultation papers, control periods, and specific financial data (like CAPEX, Opex, Traffic) for airport operators such as DIAL, MIAL, BIAL, HIAL.**
Your primary task is to take a user's complex question related to these topics and break it down into a series of 1 to 20 simple, self-contained search queries that can be individually executed against a document index. Each search query should aim to find a specific piece of information (e.g., a specific figure, a justification, a comparison point) needed to answer the overall complex question.
If the user's question is already simple and can be answered with a single search, return just that single query in the list.
If the question is very complex and might require more distinct search steps, formulate the most critical 1 to 20 search queries, focusing on distinct pieces of information.
Return your response ONLY as a JSON list of strings, where each string is a search query.

CONVERSATION HISTORY:
{history_formatted}

User's complex question: {user_question}
Your JSON list of search queries:"""
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
        "Analysis of manpower expenditure projection for BIAL for fourth control period": "The uploaded document proposes: '{document_summary}'. Use the following steps for analysing the manpower expenditure projected by BIAL...",
        "Analysis of actual manpower expenditure for BIAL for third control period": "The uploaded document proposes: '{document_summary}'. Use the following steps for analyzing the actual manpower expenditure for the third control period..."
    },
    "Utility Analysis": {
        "Analysis of electricity expenditure projection for BIAL for third control period": "The uploaded document proposes: '{document_summary}'. Use the following steps for analysing the electricity expenditure projected by BIAL...",
        "Analysis of water expenditure projection for BIAL for third control period": "The uploaded document proposes: '{document_summary}'. Use the following steps for analysing the water expenditure projected by BIAL..."
    },
    "R&M Analysis": {
        "Projected R&M Expenditure Analysis": "The uploaded document contains the following summary: '{document_summary}'. Now, perform these steps: ...",
        "Actual R&M Expenditure True-Up Analysis": "The uploaded document contains the following summary: '{document_summary}'. Now, analyze..."
    }
}

# --- API Endpoints ---
@app.post("/generate-initial-report")
async def generate_initial_report(analysis_title: str, file: UploadFile = File(...)):
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        extracted_text = extract_text_from_docx(file)
        summary_prompt = f"Summarize the key points of this document text:\n\n---\n{extracted_text[:10000]}\n---"
        summary_response = clients.synthesis_openai_client.chat.completions.create(model=config.deployment_id, messages=[{"role": "user", "content": summary_prompt}], temperature=0.1, max_tokens=1000)
        document_summary = summary_response.choices[0].message.content

        prompt_template = next((p for cat in analysis_prompts_config.values() for t, p in cat.items() if t == analysis_title), None)
        if not prompt_template:
            raise HTTPException(status_code=404, detail=f"Analysis title '{analysis_title}' not found.")
            
        full_prompt = prompt_template.format(document_summary=document_summary)
        system_prompt = "You are an AI assistant for Multiyear Tariff submission for AERA."
        analysis_answer = generate_answer_from_search(user_question=full_prompt, history=[], system_prompt=system_prompt)
        
        final_report = f"<h2>{analysis_title}</h2><h3>Summary</h3><p>{document_summary}</p><hr /><h3>Analysis</h3>{analysis_answer}"
        return {"report": final_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {e}")

@app.post("/chat", response_model=ChatResponse)
async def conversational_chat(request: ChatRequest):
    system_prompt = """You are an AI assistant for Multiyear Tariff submission for AERA. Your primary goal is to provide detailed, accurate, and well-structured answers based on the provided context from regulatory documents.
    
**Core Instructions:**
1.  **Answer from Context Only:** Do not make up any information. Your answers must be based solely on the provided context.
2.  **Detailed Responses:** Your final response should be comprehensive, aiming for at least 1500 words where the context supports it.
3.  **Source Attribution:** Always handle terminology related to the "Authority's" stance with precision. Differentiate between "approved", "decided", "proposed", and "considered" based on the exact wording in the context.
4.  **Table Referencing:** For any data, figures, or claims extracted from a table, you MUST cite the corresponding table number in your response.

**Formatting Rules (CRITICAL):**
* **HTML Tables:** Any data that is tabular in nature MUST be formatted as a proper HTML table. Use `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, and `<td>` tags. **Do not use plain text, markdown, or any other format for tables.**
* **Bullet Points:** Use bullet points (with `<ul>` and `<li>` tags) for lists, summaries, or justifications to improve readability.
* **Headings:** Use HTML heading tags (`<h2>`, `<h3>`) to structure your response with clear sections.
* **No Raw Calculations:** Perform all mathematical calculations internally. Do NOT show the formulas or step-by-step calculations in your final response. Present only the final results and conclusions in a clear, narrative text or within a table.
"""
    answer = generate_answer_from_search(user_question=request.question, history=request.history, system_prompt=system_prompt)
    return ChatResponse(answer=answer)

@app.post("/mda-chat")
async def mda_chat(request: ChatRequest):
    system_prompt = "You are an expert AI assistant for AERA regulatory matters... Your response should be at least 500 words."
    answer = generate_answer_from_search(user_question=request.question, history=request.history, system_prompt=system_prompt)
    return {"answer": answer}

@app.post("/refine-report")
async def refine_report(request: RefineReportRequest):
    refinement_prompt = f"Integrate the NEW INFORMATION into the ORIGINAL REPORT seamlessly...\n\nORIGINAL REPORT:\n{request.original_report}\n\nNEW INFORMATION:\n{request.new_info}\n\nREFINED REPORT:"
    try:
        response = clients.synthesis_openai_client.chat.completions.create(model=config.deployment_id, messages=[{"role": "user", "content": refinement_prompt}], temperature=0.2, max_tokens=4000)
        return {"refined_report": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during report refinement: {e}")

@app.post("/feedback")
async def handle_feedback(request: FeedbackRequest):
    if not clients.mongo_client:
        raise HTTPException(status_code=500, detail="Database connection not configured.")
    try:
        feedback_collection = clients.mongo_client[config.cosmos_database_name][config.cosmos_feedback_collection]
        feedback_collection.insert_one(request.dict())
        return {"status": "success", "message": "Feedback recorded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}")

@app.post("/download-report")
async def download_report(request: DownloadRequest):
    try:
        html_content = request.html_content
        soup = BeautifulSoup(html_content, 'lxml')
        document = docx.Document()

        for element in soup.find_all(['h2', 'h3', 'p', 'table', 'ul']):
            if element.name == 'h2':
                document.add_heading(element.get_text(), level=2)
            elif element.name == 'h3':
                document.add_heading(element.get_text(), level=3)
            elif element.name == 'p':
                document.add_paragraph(element.get_text())
            elif element.name == 'ul':
                for li in element.find_all('li'):
                    document.add_paragraph(li.get_text(), style='List Bullet')
            elif element.name == 'table':
                headers = [th.get_text() for th in element.find_all('th')]
                table = document.add_table(rows=1, cols=len(headers))
                table.style = 'Table Grid'
                hdr_cells = table.rows[0].cells
                for i, header in enumerate(headers):
                    hdr_cells[i].text = header

                for row in element.find_all('tr')[1:]: # Skip header row
                    row_cells_data = [td.get_text() for td in row.find_all('td')]
                    row_cells = table.add_row().cells
                    for i, cell_data in enumerate(row_cells_data):
                        row_cells[i].text = cell_data
        
        # Use a temporary file to avoid saving to disk in a serverless environment
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            document.save(tmp.name)
            return FileResponse(
                path=tmp.name, 
                filename="BIAL_Report.docx", 
                media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Word document: {e}")

@app.get("/")
def read_root():
    return {"message": "BIAL Regulatory Platform API is running."}

# To run this app: uvicorn main:app --reload