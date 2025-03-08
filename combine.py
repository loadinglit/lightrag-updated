# import os
# import asyncio
# from fastapi import FastAPI, UploadFile, Form, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from lightrag import LightRAG, QueryParam
# from lightrag.llm import gpt_4o_mini_complete

# # FastAPI app initialization
# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust for allowed origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# WORKING_DIR = "./dickens"
# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)

# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=gpt_4o_mini_complete,  # LLM model
# )

# @app.post("/ingest")
# async def ingest_document(file: UploadFile):
#     """
#     Endpoint to upload and ingest a document.
#     """
#     if not file:
#         raise HTTPException(status_code=400, detail="No file uploaded.")

#     file_path = os.path.join(WORKING_DIR, file.filename)

#     try:
#         # Save uploaded file
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
        
#         # Read and ingest content
#         with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#             content = f.read()

#         await rag.ainsert(content)
#         return JSONResponse({"message": f"Document {file.filename} ingested successfully!"})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")


# @app.post("/query")
# async def query_document(query: str = Form(...), search_type: str = Form("naive")):
#     """
#     Endpoint to query the ingested documents.
#     """
#     if not query:
#         raise HTTPException(status_code=400, detail="Query text is required.")

#     try:
#         result = await rag.aquery(query, param=QueryParam(mode=search_type))
#         return JSONResponse({search_type: result})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")


# # Run the app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


import os
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import aiofiles
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize Chat Model
chat_model = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    deployment_name=deployment_name,
)

#----------------------------------------------------------------------------------
# Custom Azure OpenAI completion function for LightRAG

async def azure_openai_acomplete(messages: list, hashing_kv: str = None, history_messages: list = None) -> dict:
    """
    Async completion function using Azure OpenAI for LightRAG.
    Args:
        messages: String or list of message dictionaries
        hashing_kv: Optional hashing key-value parameter required by LightRAG
        history_messages: Optional message history required by LightRAG
    Returns:
        dict: Response in the format expected by LightRAG
    """
    try:
        # Handle the case where messages is a string
        if isinstance(messages, str):
            message_content = messages
        else:
            # Handle the case where messages is a list of dictionaries
            message_content = messages[0] if isinstance(messages, list) else messages

        # Create the message for Azure OpenAI
        langchain_messages = [HumanMessage(content=message_content)]

        # Append history messages if they exist
        if history_messages:
            for msg in history_messages:
                if isinstance(msg, str):
                    langchain_messages.append(HumanMessage(content=msg))
                elif isinstance(msg, dict) and "content" in msg:
                    langchain_messages.append(HumanMessage(content=msg["content"]))

        # Get completion from Azure OpenAI
        response = await chat_model.ainvoke(langchain_messages)
        
        # Format response to match what LightRAG expects
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.content
                }
            }]
        }
    except Exception as e:
        logger.error(f"Error in Azure OpenAI completion: {str(e)}")
        raise
#-------------------------------------------------------------------------------------------------------------------


# FastAPI app initialization
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG Configuration
WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    # llm_model_func=gpt_4o_mini_complete,
    llm_model_func=azure_openai_acomplete,
)

async def process_csv_to_text(file: UploadFile) -> str:
    """
    Process CSV file and convert to summarized text.
    Returns the path to the generated text file.
    """
    # Save uploaded file
    file_location = f"uploaded_files/{file.filename}"
    os.makedirs("uploaded_files", exist_ok=True)
    async with aiofiles.open(file_location, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Load CSV and process data
    data = pd.read_csv(file_location, encoding='utf-8')
    headers = data.columns.tolist()
    results = []

    for index, row in data.iterrows():
        row_data = row.tolist()
        row_data = [str(val) if pd.notna(val) else "" for val in row_data]
        combined_text = ". ".join([f"{header}: {row_data[i]}" for i, header in enumerate(headers)])

        message = HumanMessage(content=(
            f"Summarize the following row of data into a concise paragraph without line breaks. "
            f"Each row should have a distinct summary that reflects its specific details in a single paragraph. "
            f"Provide a narrative that integrates all key elements from this row: {combined_text}."
        ))

        response = chat_model.invoke([message])
        summary = response.content.strip()
        results.append(f"{index + 1}: {summary}\n")

    # Save results to a text file
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, "extract.txt")
    with open(output_file_path, "w", encoding='utf-8') as f:
        for summary in results:
            f.write(summary + "\n\n")

    return output_file_path

@app.post("/ingest")
async def ingest_document(file: UploadFile):
    """
    Endpoint to process and ingest a document.
    Handles both CSV and TXT files.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        if file.filename.endswith('.csv'):
            # Process CSV to text first
            text_file_path = await process_csv_to_text(file)
            
            # Read the generated text file and ingest it
            with open(text_file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            # For direct text file ingestion
            content = await file.read()
            content = content.decode("utf-8")

        # Ingest the content
        await rag.ainsert(content)
        return JSONResponse({
            "message": f"Document {file.filename} processed and ingested successfully!",
            "text_file": text_file_path if file.filename.endswith('.csv') else None
        })

    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")

@app.post("/query")
async def query_document(query: str = Form(...), search_type: str = Form("hybrid")):
    """
    Endpoint to query the ingested documents.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required.")

    try:
        result = await rag.aquery(query, param=QueryParam(mode=search_type))
        return JSONResponse({search_type: result})
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
