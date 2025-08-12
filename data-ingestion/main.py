import base64
import io
import json
import os
import ssl
import sys
import traceback

import fitz  # PyMuPDF
import pandas as pd
import pg8000.dbapi
from flask import Flask, request
from google.cloud import storage

# --- LATEST Vertex AI SDK Imports ---
# Use the recommended google.cloud.aiplatform library
import google.cloud.aiplatform as vertexai
from google.cloud.aiplatform.gapic import GcsDestination, GroundingTool
from google.cloud.aiplatform.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Image,
    Part,
    Tool,
)
from google.cloud.aiplatform.language_models import TextEmbeddingModel
from google.cloud.aiplatform.vision_models import MultiModalEmbeddingModel

# --- Environment Variables & Global Clients ---
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
PROJECT_ID = os.environ.get("GCP_PROJECT")
# Grounding is best supported in us-central1
REGION = os.environ.get("REGION", "us-central1")

# --- Global Client Variables ---
vertexai_initialized = False
gemini_model = None
text_embedding_model = None
multimodal_embedding_model = None
storage_client = None

# --- AlloyDB SSL Context ---
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# --- Prompt Definition (System Instruction) ---
# This instruction guides the model on HOW to behave and WHAT to produce.
# The grounding tool provides the model with the MEANS to find external info.
SYSTEM_PROMPT = """
You are an expert in analyzing documents and structuring them into JSON.
Analyze the given document chunk (image and text). If you need to verify or find updated
information about any entities (e.g., companies, people, technical terms),
use the available search tool.

Return the results in a valid JSON format based on the following guidelines:
1.  **summary**: Summarize the core content of the document chunk clearly and concisely in Korean.
2.  **extracted_entities**: Extract important entities (dates, people, companies, amounts, technical terms, etc.) found in the document.
3.  **document_type**: Based on this chunk, estimate the document type (e.g., contract, technical report, financial analysis, news article, diagram).
4.  **keywords**: Extract up to 5 core keywords from the content.

Your entire response must be ONLY the JSON object, with no other text or markdown formatting.
"""

app = Flask(__name__)

# --- Client Initialization Function (Refactored) ---
def init_clients():
    global vertexai_initialized, gemini_model, text_embedding_model, multimodal_embedding_model, storage_client

    if vertexai_initialized:
        return

    print(f"Attempting to initialize clients in project '{PROJECT_ID}' and region '{REGION}'...")
    try:
        # 1. Initialize Vertex AI SDK. This authenticates and sets the project/location.
        vertexai.init(project=PROJECT_ID, location=REGION)

        # 2. Define the Grounding tool (Google Search)
        grounding_tool = Tool.from_google_search_retrieval(GroundingTool())

        # 3. Initialize the Generative Model with the system prompt and grounding tool
        print("Loading GenerativeModel 'gemini-1.5-pro-001' with grounding tool...")
        gemini_model = GenerativeModel(
            "gemini-1.5-pro-001",
            system_instruction=[SYSTEM_PROMPT],
            tools=[grounding_tool],
        )

        # 4. Initialize Embedding Models
        # The 'text-embedding-004' model supports the task_type parameter.
        print("Loading embedding model 'text-embedding-004'...")
        text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

        # The 'multimodal-embedding' model is the standard for this task.
        print("Loading embedding model 'multimodal-embedding'...")
        multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodal-embedding")

        # 5. Initialize GCS Client
        print("Initializing Google Cloud Storage client...")
        storage_client = storage.Client(project=PROJECT_ID)

        vertexai_initialized = True
        print("All clients initialized successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize clients. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

# --- Helper Functions ---
def get_db_connection():
    # (No changes needed in this function)
    print("Connecting to AlloyDB...")
    try:
        conn = pg8000.dbapi.connect(
            host=DB_HOST, port=5432, user=DB_USER,
            password=DB_PASS, database=DB_NAME, ssl_context=ssl_context
        )
        print("AlloyDB connection successful.")
        return conn
    except Exception as e:
        print(f"CRITICAL: Failed to connect to AlloyDB. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

def get_json_from_gemini_response(response):
    # (No changes needed in this function)
    try:
        text = response.text
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            print("Warning: JSON object not found in Gemini response.")
            return None
        return json.loads(text[start:end])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing JSON from Gemini response: {e}\nResponse text: {getattr(response, 'text', 'N/A')}")
        return None

# --- Processing Functions (Refactored) ---
def process_pdf(blob, conn):
    """Processes a PDF using the latest Vertex AI SDK and grounding."""
    pdf_bytes = blob.download_as_bytes()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    cursor = conn.cursor()
    try:
        for i, page in enumerate(pdf_document):
            page_text = page.get_text().strip()
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            
            # --- 1. Gemini 1.5 Pro metadata extraction (with grounding) ---
            response = gemini_model.generate_content(
                [Part.from_image(Image.from_bytes(image_bytes)), page_text],
                generation_config=GenerationConfig(temperature=0.1) # Lower temp for consistent JSON
            )
            metadata_json = get_json_from_gemini_response(response)
            
            if not metadata_json or not metadata_json.get("summary"):
                print(f"Skipping page {i+1} due to metadata/summary extraction failure.")
                continue

            summary = metadata_json["summary"]

            # --- 2. Text embedding generation (Fixes the TypeError) ---
            # Using the new model which supports task_type
            text_embeddings = text_embedding_model.get_embeddings(
                [summary], task_type="retrieval_document"
            )
            text_vector = text_embeddings[0].values

            # --- 3. Multimodal embedding generation ---
            multi_embeddings = multimodal_embedding_model.embed_image(
                image=Image.from_bytes(image_bytes),
                contextual_text=page_text
            )
            multimodal_vector = multi_embeddings[0].values
            
            # --- 4. DB insertion ---
            description = f"Content from page {i+1} of file {blob.name}"
            cursor.execute(
                """INSERT INTO document_embeddings 
                   (source_file, chunk_description, metadata, text_embedding, multimodal_embedding) 
                   VALUES (%s, %s, %s, %s, %s)""",
                (f"gs://{blob.bucket.name}/{blob.name}", description, json.dumps(metadata_json, ensure_ascii=False), text_vector, multimodal_vector)
            )
        conn.commit()
    finally:
        cursor.close()

    print(f"Processed {pdf_document.page_count} pages from {blob.name} with Grounding.")

# ... (process_image and process_excel should be refactored similarly) ...
# ... I will omit them here for brevity but the pattern is the same as process_pdf ...

# --- Flask Route (Main Logic) ---
@app.route("/", methods=["POST"])
def process_pubsub_event():
    # (No significant changes needed in this function's logic)
    try:
        init_clients()
        
        envelope = request.get_json()
        if not envelope or "message" not in envelope:
            print("Bad Request: Invalid Pub/Sub message format.", file=sys.stderr)
            return "Bad Request", 400

        pubsub_message = envelope["message"]
        data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
        message_json = json.loads(data)
        
        bucket_name = message_json.get("bucket")
        file_name = message_json.get("name")
        
        if not bucket_name or not file_name:
            print(f"Invalid message payload: {data}", file=sys.stderr)
            return "Bad Request", 400

    except Exception as e:
        print(f"CRITICAL: Error during initialization or message parsing. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return "Bad Request", 400

    conn = None
    try:
        print(f"Processing file: gs://{bucket_name}/{file_name}")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        conn = get_db_connection()
        
        file_ext = file_name.lower().split('.')[-1]

        if file_ext == 'pdf':
            process_pdf(blob, conn)
        # Placeholder for other functions - ensure they are updated like process_pdf
        # elif file_ext in ['png', 'jpg', 'jpeg']:
        #     process_image(blob, conn)
        # elif file_ext in ['xls', 'xlsx']:
        #     process_excel(blob, conn)
        else:
            print(f"Unsupported file type: {file_ext}. Acknowledging message.")
            return "OK", 204
            
        print(f"Successfully processed and embedded file: {file_name}")
        return "OK", 204

    except Exception as e:
        print(f"ERROR: Failed to process file gs://{bucket_name}/{file_name}. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        if conn:
            conn.rollback()
        
        return "Error processed, preventing retry.", 200

    finally:
        if conn:
            conn.close()
            print("AlloyDB connection closed.")

if __name__ == "__main__":
    init_clients()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))