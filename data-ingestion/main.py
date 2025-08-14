import base64
import io
import json
import os
import ssl
import sys
import traceback

import fitz
import pandas as pd
import pg8000.dbapi
from flask import Flask, request

# --- Environment Variables & Global Placeholders ---
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
PROJECT_ID = os.environ.get("GCP_PROJECT")
REGION = os.environ.get("REGION", "asia-northeast3") # Set to your actual region

# --- Global Client Variables (populated by init_clients) ---
vertexai_initialized = False
gemini_model = None
text_embedding_model = None
multimodal_embedding_model = None
storage_client = None

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

SYSTEM_PROMPT = """
You are an expert in analyzing documents and structuring them into JSON.
Analyze the given document chunk (image and text) and return the results in a
valid JSON format based on the following guidelines:
1.  **summary**: Summarize the core content of the document chunk clearly and concisely in Korean.
2.  **extracted_entities**: Extract important entities (dates, people, companies, amounts, technical terms, etc.) found in the document.
3.  **document_type**: Based on this chunk, estimate the document type (e.g., contract, technical report, financial analysis, news article, diagram).
4.  **keywords**: Extract up to 5 core keywords from the content.
Your entire response must be ONLY the JSON object, with no other text or markdown formatting.
"""

app = Flask(__name__)

def init_clients():
    global vertexai_initialized, gemini_model, text_embedding_model, multimodal_embedding_model, storage_client

    if vertexai_initialized:
        return

    print(f"Attempting to initialize clients in project '{PROJECT_ID}' and region '{REGION}'...")
    try:
        # These imports will now succeed because of requirements.txt
        import google.cloud.aiplatform as vertexai
        from google.cloud import storage
        from google.cloud.aiplatform.generative_models import GenerativeModel, Image, Part
        from google.cloud.aiplatform.language_models import TextEmbeddingModel
        from google.cloud.aiplatform.vision_models import MultiModalEmbeddingModel
        
        vertexai.init(project=PROJECT_ID, location=REGION)

        print("Loading GenerativeModel 'gemini-2.5-pro'...")
        gemini_model = GenerativeModel("gemini-2.5-pro", system_instruction=[SYSTEM_PROMPT])

        print("Loading embedding model 'text-multilingual-embedding-002'...")
        text_embedding_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
        
        print("Loading embedding model 'multimodal-embedding'...")
        multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodal-embedding")

        print("Initializing Google Cloud Storage client...")
        storage_client = storage.Client(project=PROJECT_ID)

        vertexai_initialized = True
        print("All clients initialized successfully.")

    except Exception as e:
        print(f"CRITICAL: Failed to initialize clients. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

def get_db_connection():
    # ... (implementation from previous steps is fine)
    print("Connecting to AlloyDB...")
    try:
        conn = pg8000.dbapi.connect(host=DB_HOST, port=5432, user=DB_USER, password=DB_PASS, database=DB_NAME, ssl_context=ssl_context)
        print("AlloyDB connection successful.")
        return conn
    except Exception as e:
        print(f"CRITICAL: Failed to connect to AlloyDB. Error: {e}", file=sys.stderr); traceback.print_exc(); raise

def get_json_from_gemini_response(response):
    # ... (implementation from previous steps is fine)
    try:
        text = response.text; start = text.find('{'); end = text.rfind('}') + 1
        if start == -1 or end == 0: return None
        return json.loads(text[start:end])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing JSON: {e}\nResponse: {getattr(response, 'text', 'N/A')}"); return None

def process_pdf(blob, conn):
    from google.cloud.aiplatform.generative_models import Image, Part

    pdf_bytes = blob.download_as_bytes()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    cursor = conn.cursor()
    try:
        for i, page in enumerate(pdf_document):
            page_text = page.get_text().strip()
            pix = page.get_pixmap(dpi=150); image_bytes = pix.tobytes("png")
            
            response = gemini_model.generate_content([Part.from_image(Image.from_bytes(image_bytes)), page_text])
            metadata_json = get_json_from_gemini_response(response)
            
            if not metadata_json or not metadata_json.get("summary"): continue
            summary = metadata_json["summary"]

            text_embeddings = text_embedding_model.get_embeddings([summary]); text_vector = text_embeddings[0].values
            multi_embeddings = multimodal_embedding_model.embed_image(image=Image.from_bytes(image_bytes), contextual_text=page_text); multimodal_vector = multi_embeddings[0].values
            
            description = f"Content from page {i+1} of file {blob.name}"
            cursor.execute("""INSERT INTO document_embeddings (source_file, chunk_description, metadata, text_embedding, multimodal_embedding) VALUES (%s, %s, %s, %s, %s)""",(f"gs://{blob.bucket.name}/{blob.name}", description, json.dumps(metadata_json, ensure_ascii=False), text_vector, multimodal_vector))
        conn.commit()
    finally:
        cursor.close()
    print(f"Processed {pdf_document.page_count} pages from {blob.name}.")

@app.route("/", methods=["POST"])
def process_pubsub_event():
    try:
        init_clients()
        
        envelope = request.get_json();
        if not envelope or "message" not in envelope: return "Bad Request", 400
        pubsub_message = envelope["message"]
        data = base64.b64decode(pubsub_message["data"]).decode("utf-8"); message_json = json.loads(data)
        bucket_name = message_json.get("bucket"); file_name = message_json.get("name")
        if not bucket_name or not file_name: return "Bad Request", 400
    except Exception as e:
        print(f"CRITICAL: Init/parse error. Error: {e}", file=sys.stderr); traceback.print_exc(); return "Bad Request", 400

    conn = None
    try:
        print(f"Processing file: gs://{bucket_name}/{file_name}")
        bucket = storage_client.bucket(bucket_name); blob = bucket.blob(file_name)
        conn = get_db_connection()
        file_ext = file_name.lower().split('.')[-1]

        if file_ext == 'pdf': process_pdf(blob, conn)
        else: return "OK", 204
        
        print(f"Successfully processed and embedded file: {file_name}"); return "OK", 204
    except Exception as e:
        print(f"ERROR: Failed to process file gs://{bucket_name}/{file_name}. Error: {e}", file=sys.stderr); traceback.print_exc()
        if conn: conn.rollback()
        return "Error processed, preventing retry.", 200
    finally:
        if conn: conn.close(); print("AlloyDB connection closed.")

if __name__ == "__main__":
    init_clients()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))