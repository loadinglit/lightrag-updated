import streamlit as st
import numpy as np
import os
import tempfile
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import pandas as pd
from langchain_openai import AzureChatOpenAI
from typing import Tuple, Union, Dict, Any
from dotenv import load_dotenv
import logging
import aiofiles
from openai import AzureOpenAI
import textract
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment Variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

WORKING_DIR = "./dickens"

async def llm_model_func(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    keyword_extraction=False, 
    **kwargs
) -> str:
    """
    Async function to generate completions using Azure OpenAI
    """
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Create chat completion
    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
        timeout=30,
    )
    
    return chat_completion.choices[0].message.content

async def embedding_func(texts: list[str]) -> np.ndarray:
    """
    Async function to generate embeddings using Azure OpenAI
    """
    client = AzureOpenAI(
        api_key=AZURE_EMBEDDING_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    )
    
    embedding = client.embeddings.create(
        model=AZURE_EMBEDDING_DEPLOYMENT, 
        input=texts
    )
    
    embeddings = [item.embedding for item in embedding.data]
    return np.array(embeddings)

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Initialize LightRAG with Azure OpenAI
@st.cache_resource
def initialize_rag():
    embedding_dimension = 1536
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    return rag

# Initialize the RAG system
rag = initialize_rag()

# Improvements to document processing functions
async def process_csv_to_text(file) -> str:
    """
    Process CSV file and convert to summarized text.
    """
    # Create upload and output directories
    os.makedirs("uploaded_files", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Save uploaded file
    file_location = os.path.join("uploaded_files", file.name)
    with open(file_location, "wb") as f:
        f.write(file.getvalue())

    # Load CSV and process data
    try:
        data = pd.read_csv(file_location, encoding='utf-8')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    headers = data.columns.tolist()
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, row in data.iterrows():
        status_text.text(f"Processing row {index + 1} of {len(data)}")
        progress_bar.progress((index + 1) / len(data))
        
        row_data = row.tolist()
        row_data = [str(val) if pd.notna(val) else "" for val in row_data]
        combined_text = ". ".join([f"{header}: {row_data[i]}" for i, header in enumerate(headers)])

        try:
            summary = await llm_model_func(
                prompt=(
                    f"Summarize the following row of data into a concise paragraph without line breaks. "
                    f"Each row should have a distinct summary that reflects its specific details in a single paragraph. "
                    f"Provide a narrative that integrates all key elements from this row: {combined_text}."
                )
            )
            summary = summary.strip()
            results.append(f"{index + 1}: {summary}\n")
        except Exception as e:
            st.error(f"Error processing row {index + 1}: {e}")
            continue

    # Save results to a text file
    output_file_path = os.path.join("output", "extract.txt")
    with open(output_file_path, "w", encoding='utf-8') as f:
        for summary in results:
            f.write(summary + "\n\n")

    status_text.text("Processing complete!")
    progress_bar.empty()
    
    return output_file_path


async def process_json_file(file_path: str) -> Tuple[bool, Union[str, Dict[str, Any]]]:
    """
    Process JSON file and return its contents.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return True, content
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return False, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        logger.error(f"Error processing JSON file: {str(e)}")
        return False, f"Error processing JSON file: {str(e)}"

async def process_document_file(file_path: str) -> Tuple[bool, str]:
    """
    Process non-JSON documents using textract.
    """
    try:
        text_content = textract.process(file_path)
        if not text_content:
            return False, "No content extracted from file"
        return True, text_content.decode('utf-8')
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        return False, f"Error processing document: {str(e)}"

async def ingest_document(file) -> Tuple[bool, str]:
    """
    Process and ingest a document with support for JSON and other file types.
    """
    if file is None:
        return False, "No file uploaded"

    try:
        # Create temporary directory for file processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            # Save uploaded file
            # Save uploaded file
            with open(temp_file_path, 'wb') as f:
                f.write(file.getvalue())

            # Determine file type and process accordingly
            is_json = file.filename.lower().endswith('.json')
            
            if is_json:
                success, content = await process_json_file(temp_file_path)
                if not success:
                    return False, content
                
                # Handle JSON content with retries
                retries = 0
                max_retries = 3
                while retries < max_retries:
                    try:
                        await rag.ainsert(content)
                        break
                    except Exception as e:
                        retries += 1
                        logger.error(f"JSON insertion retry {retries}/{max_retries}: {str(e)}")
                        await asyncio.sleep(10)
                
                if retries == max_retries:
                    return False, "Failed to insert JSON content after maximum retries"
            else:
                # Process other document types
                success, content = await process_document_file(temp_file_path)
                if not success:
                    return False, content
                
                # Insert processed content
                try:
                    await rag.ainsert(content)
                except Exception as e:
                    logger.error(f"Document insertion error: {str(e)}")
                    return False, f"Error inserting document content: {str(e)}"

            return True, f"Document {file.filename} processed and ingested successfully!"

    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        return False, f"Error during ingestion: {str(e)}"

async def process_query(query_text: str, search_type: str = "hybrid"):
    """Process query with proper async handling."""
    try:
        if not query_text:
            return None, "Query cannot be empty"

        # Create query parameters
        query_params = QueryParam(mode=search_type)
        
        # Execute query
        try:
            result = await rag.aquery(query_text, param=query_params)
            if result is None:
                return None, "No results found"
            return result, None
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            return None, f"Query execution failed: {str(e)}"
            
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        return None, f"Query processing failed: {str(e)}"

def main():
    st.title("LightRAG - Query System")

    # File upload section
    st.header("Document Ingestion")
    supported_types = ['txt', 'pdf', 'doc', 'docx', 'odt', 'xlsx', 'csv']
    uploaded_file = st.file_uploader("Choose a file", type=supported_types)

    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Processing document..."):
                try:
                    # Run ingestion in a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success, message = loop.run_until_complete(ingest_document(uploaded_file))
                    loop.close()
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    logger.error(f"Detailed error: {str(e)}", exc_info=True)

    # Query section
    st.header("Query Documents")
    query_text = st.text_input("Enter your query")
    search_type = st.selectbox("Search Type", ["hybrid", "local", "global"])

    if query_text and st.button("Submit Query"):
        with st.spinner("Processing query..."):
            try:
                # Run query in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result, error = loop.run_until_complete(process_query(query_text, search_type))
                loop.close()
                
                if error:
                    st.error(error)
                else:
                    st.subheader("Query Result")
                    if isinstance(result, dict):
                        st.json(result)
                    elif isinstance(result, list):
                        for idx, item in enumerate(result, 1):
                            st.write(f"Result {idx}:")
                            st.json(item) if isinstance(item, dict) else st.write(item)
                    else:
                        st.write(result)
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logger.error(f"Detailed error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()