# import streamlit as st
# import numpy as np
# import os
# import asyncio
# from lightrag import LightRAG, QueryParam
# from lightrag.utils import EmbeddingFunc
# import pandas as pd
# from langchain_core.messages import HumanMessage
# from langchain_openai import AzureChatOpenAI
# from dotenv import load_dotenv
# import logging
# import aiofiles
# from openai import AzureOpenAI

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # # Azure OpenAI Configuration
# # api_key = os.getenv("AZURE_OPENAI_API_KEY")
# # azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# # deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
# # api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# # # Initialize Chat Model
# # chat_model = AzureChatOpenAI(
# #     api_key=api_key,
# #     azure_endpoint=azure_endpoint,
# #     api_version=api_version,
# #     deployment_name=deployment_name,
# # )

# # Assuming these are defined elsewhere in your configuration
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
# AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
# AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
# AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
# WORKING_DIR = "./dickens"

# async def llm_model_func(
#     prompt, 
#     system_prompt=None, 
#     history_messages=[], 
#     keyword_extraction=False, 
#     **kwargs
# ) -> str:
#     """
#     Async function to generate completions using Azure OpenAI
    
#     Args:
#         prompt (str): The user's prompt
#         system_prompt (str, optional): System-level instructions
#         history_messages (list, optional): Previous conversation context
#         keyword_extraction (bool, optional): Flag for keyword extraction
#         **kwargs: Additional parameters like temperature, top_p, etc.
    
#     Returns:
#         str: AI-generated response
#     """
#     client = AzureOpenAI(
#         api_key=AZURE_OPENAI_API_KEY,
#         api_version=AZURE_OPENAI_API_VERSION,
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     )
    
#     # Prepare messages
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     if history_messages:
#         messages.extend(history_messages)
#     messages.append({"role": "user", "content": prompt})
    
#     # Create chat completion
#     chat_completion = client.chat.completions.create(
#         model=AZURE_OPENAI_DEPLOYMENT,
#         messages=messages,
#         temperature=kwargs.get("temperature", 0),
#         top_p=kwargs.get("top_p", 1),
#         n=kwargs.get("n", 1),
#     )
    
#     return chat_completion.choices[0].message.content

# async def embedding_func(texts: list[str]) -> np.ndarray:
#     """
#     Async function to generate embeddings using Azure OpenAI
    
#     Args:
#         texts (list[str]): List of texts to embed
    
#     Returns:
#         np.ndarray: Array of embeddings
#     """
#     client = AzureOpenAI(
#         api_key=AZURE_EMBEDDING_API_KEY,
#         api_version=AZURE_EMBEDDING_API_VERSION,
#         azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
#     )
    
#     embedding = client.embeddings.create(
#         model=AZURE_EMBEDDING_DEPLOYMENT, 
#         input=texts
#     )
    
#     embeddings = [item.embedding for item in embedding.data]
#     return np.array(embeddings)

# # RAG Configuration
# WORKING_DIR = "./dickens"
# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)

# # Initialize LightRAG with Azure OpenAI
# @st.cache_resource
# def initialize_rag():
#     embedding_dimension = 1536
#     return LightRAG(
#         working_dir=WORKING_DIR,
#         llm_model_func=llm_model_func,
#         embedding_func=EmbeddingFunc(
#         embedding_dim=embedding_dimension,
#         max_token_size=8192,
#         func=embedding_func,
#     ),
#     )
# rag = initialize_rag()

# async def process_csv_to_text(file) -> str:
#     """
#     Process CSV file and convert to summarized text.
#     Returns the path to the generated text file.
#     """
#     # Save uploaded file
#     file_location = f"uploaded_files/{file.name}"
#     os.makedirs("uploaded_files", exist_ok=True)
    
#     # Save the uploaded file
#     with open(file_location, "wb") as f:
#         f.write(file.getvalue())

#     # Load CSV and process data
#     data = pd.read_csv(file_location, encoding='utf-8')
#     headers = data.columns.tolist()
#     results = []

#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     for index, row in data.iterrows():
#         status_text.text(f"Processing row {index + 1} of {len(data)}")
#         progress_bar.progress((index + 1) / len(data))
        
#         row_data = row.tolist()
#         row_data = [str(val) if pd.notna(val) else "" for val in row_data]
#         combined_text = ". ".join([f"{header}: {row_data[i]}" for i, header in enumerate(headers)])

#         # message = HumanMessage(content=(
#         #     f"Summarize the following row of data into a concise paragraph without line breaks. "
#         #     f"Each row should have a distinct summary that reflects its specific details in a single paragraph. "
#         #     f"Provide a narrative that integrates all key elements from this row: {combined_text}."
#         # ))

#         # response = chat_model.invoke([message])
#         # summary = response.content.strip()
#         # results.append(f"{index + 1}: {summary}\n")

#                 # Replace chat_model.invoke with llm_model_func
#         summary = await llm_model_func(
#             prompt=(
#                 f"Summarize the following row of data into a concise paragraph without line breaks. "
#                 f"Each row should have a distinct summary that reflects its specific details in a single paragraph. "
#                 f"Provide a narrative that integrates all key elements from this row: {combined_text}."
#             )
#         )
#         summary = summary.strip()
#         results.append(f"{index + 1}: {summary}\n")

#     # Save results to a text file
#     output_folder = "output"
#     os.makedirs(output_folder, exist_ok=True)
#     output_file_path = os.path.join(output_folder, "extract.txt")
#     with open(output_file_path, "w", encoding='utf-8') as f:
#         for summary in results:
#             f.write(summary + "\n\n")

#     status_text.text("Processing complete!")
#     progress_bar.empty()
    
#     return output_file_path

# async def ingest_document(file):
#     """Process and ingest a document."""
#     try:
#         if file.name.endswith('.csv'):
#             # Process CSV to text first
#             text_file_path = await process_csv_to_text(file)
            
#             # Read the generated text file and ingest it
#             with open(text_file_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#         else:
#             # For direct text file ingestion
#             content = file.getvalue().decode("utf-8")

#         # Ingest the content
#         await rag.ainsert(content)
#         return True, f"Document {file.name} processed and ingested successfully!"
#     except Exception as e:
#         logger.error(f"Error during ingestion: {str(e)}")
#         return False, f"Error during ingestion: {str(e)}"

# async def query_document(query: str, search_type: str):
#     try:
#         result = await rag.aquery(query, param=QueryParam(mode=search_type))
        
#         if not result:
#             raise ValueError("Backend returned no data.")
        
#         logger.debug(f"Raw response from backend: {result}")
#         return True, result
#     except Exception as e:
#         logger.error(f"Error during query: {e}")
#         return False, f"Error during query: {e}"



# # Streamlit Interface
# st.title("Document Query System")

# # File upload section
# st.header("Document Ingestion")
# uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])

# if uploaded_file is not None:
#     if st.button("Ingest Document"):
#         with st.spinner("Processing document..."):
#             success, message = asyncio.run(ingest_document(uploaded_file))
#             if success:
#                 st.success(message)
#             else:
#                 st.error(message)

# # Query section
# st.header("Query Documents")
# query_text = st.text_input("Enter your query")
# search_type = st.selectbox("Search Type", ["hybrid", "naive"])

# if query_text and st.button("Submit Query"):
#     with st.spinner("Processing query..."):
#         result = asyncio.run(rag.aquery(query_text, param=QueryParam(mode="naive")))
#         st.subheader("Query Result")
#         st.write(result)


# # Optional: Add a health check indicator
# if st.sidebar.button("Check System Status"):
#     st.sidebar.success("System is healthy")


import streamlit as st
import numpy as np
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import pandas as pd
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import logging
import aiofiles
from openai import AzureOpenAI

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

async def ingest_document(file):
    """Process and ingest a document."""
    try:
        if file.name.endswith('.csv'):
            # Process CSV to text first
            text_file_path = await process_csv_to_text(file)
            
            # Check if file was processed successfully
            if not text_file_path:
                return False, f"Failed to process CSV file {file.name}"
            
            # Read the generated text file and ingest it
            with open(text_file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            # For direct text file ingestion
            content = file.getvalue().decode("utf-8")

        # Ingest the content
        if not content:
            return False, "No content to ingest"

        # Use try-except to handle any potential ingestion errors
        try:
            await rag.ainsert(content)
        except Exception as e:
            logger.error(f"Detailed ingestion error: {e}")
            return False, f"Error during content insertion: {e}"

        return True, f"Document {file.name} processed and ingested successfully!"
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        return False, f"Error during ingestion: {str(e)}"

async def safe_query_document(query: str, search_type: str = "naive"):
    """
    Safely query documents with additional error handling
    """
    try:
  
        # Ensure query is not empty
        if not query:
            return False, "Query cannot be empty"

        result = await rag.aquery(query, param=QueryParam(mode=search_type))
        
        if not result:
            return False, "No results found for the given query."
        
        return True, result
    except Exception as e:
        logger.error(f"Query error: {e}")
        return False, f"An error occurred during querying: {e}"

# Streamlit Interface
def main():
    st.title("LightRAG - Query System")

    # Sidebar for system status
    st.sidebar.header("System Status")
    if st.sidebar.button("Check System Status"):
        st.sidebar.success("System is ready")

    # File upload section
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])

    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Processing document..."):
                try:
                    success, message = asyncio.run(ingest_document(uploaded_file))
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        # Query section
    st.header("Query Documents")
    query_text = st.text_input("Enter your query")
    search_type = st.selectbox("Search Type", ["local", "global", "hybrid"])

    if query_text and st.button("Submit Query"):
        with st.spinner("Processing query..."):
            try:
                # Directly use rag.aquery with the query text and search type
                result = asyncio.run(rag.aquery(query_text, param=QueryParam(mode=search_type)))
                
                st.subheader("Query Result")
                
                # More robust result display
                if isinstance(result, dict):
                    st.json(result)
                elif isinstance(result, list):
                    # Handle list of results
                    for idx, item in enumerate(result, 1):
                        st.write(f"Result {idx}:")
                        st.json(item) if isinstance(item, dict) else st.write(item)
                else:
                    st.write(result)
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                # Log full traceback
                import traceback
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()