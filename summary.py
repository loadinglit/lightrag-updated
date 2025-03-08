from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI  # Ensure correct import
from dotenv import load_dotenv
import aiofiles

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Correct variable reference
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # Deployment name from .env
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize Chat Model
chat_model = AzureChatOpenAI(
    api_key=api_key,  # Changed from openai_api_key
    azure_endpoint=azure_endpoint,  # Using azure_endpoint parameter
    api_version=api_version,
    deployment_name=deployment_name,
)
# FastAPI app
app = FastAPI()

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_location = f"uploaded_files/{file.filename}"
        os.makedirs("uploaded_files", exist_ok=True)
        async with aiofiles.open(file_location, "wb") as f:
            content = await file.read()
            await f.write(content)

        # Load CSV and process data
        data = pd.read_csv(file_location)
        headers = data.columns.tolist()
        results = []

        for index, row in data.iterrows():
            # Prepare row content
            row_data = row.tolist()
            combined_text = ". ".join([f"{header}: {row_data[i]}" for i, header in enumerate(headers)])

            # Create a HumanMessage for the prompt
            message = HumanMessage(content=(
                f"Summarize the following row of data into a concise paragraph without line breaks. "
                f"Each row should have a distinct summary that reflects its specific details in a single paragraph. "
                f"Provide a narrative that integrates all key elements from this row: {combined_text}."
            ))

            # Call Azure Chat API
            response = chat_model.invoke([message])
            summary = response.content.strip()
            results.append(f"{index + 1}: {summary}\n")

        # Save results to a text file
        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, "extract.txt")
        with open(output_file_path, "w") as f:
            for summary in results:
                f.write(summary + "\n\n")

        return JSONResponse(content={"message": "File processed successfully!", "output": output_file_path})

    except Exception as e:
        # Log error details
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)
