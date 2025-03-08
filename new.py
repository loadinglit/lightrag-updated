import streamlit as st
import numpy as np
import os
import tempfile
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import pandas as pd
from typing import Tuple, Union, Dict, Any, List
from dotenv import load_dotenv
import logging
import textract
import json
from openai import AzureOpenAI
import nest_asyncio

from visuals import Neo4jConnection, create_graph_visualization


st.set_page_config(page_title="Neo4j Graph Visualization", layout="wide")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

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

async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    try:
        chat_completion = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            top_p=kwargs.get("top_p", 1),
            n=kwargs.get("n", 1),
            timeout=30,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        raise

async def embedding_func(texts: list[str]) -> np.ndarray:
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

async def process_json_for_entities(json_content: Union[Dict, List]) -> str:
    """
    Process JSON content to create a structured text representation for entity extraction
    """
    def format_json_item(item, prefix=""):
        if isinstance(item, dict):
            return "\n".join(f"{prefix}{k}: {format_json_item(v)}" for k, v in item.items())
        elif isinstance(item, list):
            return "\n".join(format_json_item(i, prefix + "  ") for i in item)
        else:
            return str(item)

    if isinstance(json_content, list):
        formatted_text = "\n\n".join(format_json_item(item) for item in json_content)
    else:
        formatted_text = format_json_item(json_content)

    return formatted_text

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Initialize LightRAG with Azure OpenAI
@st.cache_resource
def initialize_rag():
    embedding_dimension = 1536
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

# Initialize the RAG system
rag = initialize_rag()

def run_sync(func, *args, **kwargs):
    """Helper function to run async functions in sync context"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(func(*args, **kwargs))

async def process_document_content(content: Union[str, Dict, List]) -> Tuple[bool, str]:
    """Process and insert document content"""
    try:
        if isinstance(content, (dict, list)):
            # Convert JSON to structured text for better entity extraction
            formatted_content = await process_json_for_entities(content)
            logger.debug(f"Formatted JSON content: {formatted_content[:500]}...")  # Log first 500 chars
            await rag.ainsert(formatted_content)
        else:
            await rag.ainsert(content)
        return True, "Content processed and ingested successfully!"
    except Exception as e:
        logger.error(f"Content processing error: {str(e)}")
        return False, f"Error processing content: {str(e)}"

async def ingest_document(file) -> Tuple[bool, str]:
    """Process and ingest a document with proper async handling"""
    if file is None:
        return False, "No file uploaded"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, file.name)
            
            # Save uploaded file
            with open(temp_file_path, 'wb') as f:
                f.write(file.getvalue())

            # Process file based on type
            if file.name.lower().endswith('.json'):
                try:
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        logger.debug(f"Loaded JSON content type: {type(content)}")
                        logger.debug(f"JSON content preview: {str(content)[:500]}...")
                    success, message = await process_document_content(content)
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON format: {str(e)}"
            else:
                try:
                    text_content = textract.process(temp_file_path)
                    if not text_content:
                        return False, "No content extracted from file"
                    content = text_content.decode('utf-8')
                    success, message = await process_document_content(content)
                except Exception as e:
                    return False, f"Error extracting content: {str(e)}"

            return success, message

    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        return False, f"Error during ingestion: {str(e)}"

async def process_query(query_text: str, search_type: str = "hybrid") -> Tuple[Union[dict, list, str, None], Union[str, None]]:
    """Process query with async handling"""
    if not query_text:
        return None, "Query cannot be empty"

    try:
        query_params = QueryParam(mode=search_type)
        result = await rag.aquery(query_text, param=query_params)
        return (result, None) if result is not None else (None, "No results found")
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        return None, f"Query processing failed: {str(e)}"

def main():

    st.title("LightRAG - Query System")

    # File upload section
    st.header("Document Ingestion")
    supported_types = ['txt', 'pdf', 'doc', 'docx', 'odt', 'xlsx', 'csv', 'json']
    uploaded_file = st.file_uploader("Choose a file", type=supported_types)

    if uploaded_file is not None and st.button("Ingest Document"):
        with st.spinner("Processing document..."):
            success, message = run_sync(ingest_document, uploaded_file)
            if success:
                st.success(message)
            else:
                st.error(message)
        
    # Database connection settings
 

    cypher_query = "MATCH (n)-[r]->(m) RETURN COALESCE(n.name, n.id, ID(n)) AS source, COALESCE(m.name, m.id, ID(m)) AS target,  type(r) AS relation LIMIT 173"
        

    with st.sidebar:
        st.header("Database Settings")
        neo4j_uri = st.text_input("Neo4j URI", "neo4j+s://70aaffdf.databases.neo4j.io:7687")
        neo4j_user = st.text_input("Username", "neo4j")
        neo4j_password = st.text_input("Password", "h399A3mepG_hj_tiQOPwaq9ufzAEGXeFvnqWhFrXPvQ")

        # Show graph
    if st.button("Connect and Visualize"):
            try:
                    st.title("Neo4j Graph Visualization")
                # Initialize connection
                    conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
                    # Query input

                    with st.spinner("Loading graph from Neo4j..."):
                        graph_data = conn.query(cypher_query)
                        if graph_data:
                            network = create_graph_visualization(graph_data)
                            st.write(network)
                        else:
                            st.warning("Failed to load graph from Neo4j")


                    with st.spinner("Generating visualization..."):
                        # Create and display graph
                        graph_file = create_graph_visualization(graph_data)
                        with open(graph_file, 'r', encoding='utf-8') as f:
                            html_data = f.read()
                        st.components.v1.html(html_data, height=800)
                        
                        # Cleanup temporary file
                        os.unlink(graph_file)
                                # Close connection
                    conn.close()
                
            except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        
    if st.button("Query Documents"):
            # Query section
            st.header("Query Documents")
            query_text = st.text_input("Enter your query")
            search_type = st.selectbox("Search Type", ["hybrid", "local", "global"])

            if query_text and st.button("Submit Query"):
                with st.spinner("Processing query..."):
                    result, error = run_sync(process_query, query_text, search_type)
                    
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


        





if __name__ == "__main__":
    main()