import base64, json, os, sys, traceback
from flask import Flask, request
import fitz
import google.generativeai as genai
from google.cloud import storage

# --- 환경 변수 ---
PROJECT_ID = os.environ.get("GCP_PROJECT")
MODEL_REGION = os.environ.get("REGION", "us-central1")
# 파싱된 결과를 저장할 새로운 버킷
PARSED_BUCKET_NAME = os.environ.get("PARSED_BUCKET") 

# --- Global Client Variables (populated by init_clients) ---
clients_initialized = False
gemini_model = None
storage_client = None

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
    global clients_initialized, gemini_model, storage_client
    if clients_initialized: return

    print(f"Attempting to initialize clients in project '{PROJECT_ID}'...")
    try:
        genai.configure(project=PROJECT_ID, location=MODEL_REGION)
        gemini_model = genai.GenerativeModel("publishers/google/models/gemini-2.5-pro", system_instruction=[SYSTEM_PROMPT])
        storage_client = storage.Client(project=PROJECT_ID)
        clients_initialized = True
        print("Parser clients initialized successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize clients. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

@app.route("/", methods=["POST"])
def process_upload_event():
    # ... Pub/Sub 메시지 파싱 ...
    bucket_name = message_json.get("bucket")
    file_name = message_json.get("name")
    
    # PDF 다운로드 및 파싱
    source_bucket = storage_client.bucket(bucket_name)
    pdf_blob = source_bucket.blob(file_name)
    pdf_bytes = pdf_blob.download_as_bytes()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    destination_bucket = storage_client.bucket(PARSED_BUCKET_NAME)

    for i, page in enumerate(pdf_document):
        page_num = i + 1
        page_text = page.get_text().strip()
        image_bytes = page.get_pixmap(dpi=150).tobytes("png")
        
        # Gemini 호출로 메타데이터 생성
        response = gemini_model.generate_content(
            [{"mime_type": "image/png", "data": image_bytes}, page_text],
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        metadata_json = json.loads(response.text)

        # 중간 결과 JSON 생성
        intermediate_data = {
            "source_file": f"gs://{bucket_name}/{file_name}",
            "page_num": page_num,
            "page_text": page_text,
            "image_base64": base64.b64encode(image_bytes).decode('utf-8'),
            "metadata": metadata_json
        }

        # 중간 결과를 새 버킷에 저장
        new_blob_name = f"{file_name}-page-{page_num}.json"
        new_blob = destination_bucket.blob(new_blob_name)
        new_blob.upload_from_string(
            json.dumps(intermediate_data, ensure_ascii=False),
            content_type="application/json"
        )
        print(f"  - Saved intermediate data to gs://{PARSED_BUCKET_NAME}/{new_blob_name}")
    
    return "OK", 204