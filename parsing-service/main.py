import base64
import json
import os
import sys
import traceback
from flask import Flask, request, jsonify
import fitz  # PyMuPDF
from google.cloud import storage

# --- 올바른 라이브러리 임포트 ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# --- 환경 변수 ---
PROJECT_ID = os.environ.get("GCP_PROJECT")
# 모델이 위치한 리전을 명시합니다.
MODEL_LOCATION = "us-central1"
PARSED_BUCKET_NAME = os.environ.get("PARSED_BUCKET")

# --- Global Client Variables ---
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

    print(f"Initializing clients with Vertex AI SDK...")
    try:
        # --- Vertex AI SDK 초기화 ---
        # 이 함수가 Cloud Run 환경의 프로젝트, 리전, 인증 정보를 모두 자동으로 처리합니다.
        vertexai.init(project=PROJECT_ID, location=MODEL_LOCATION)
        
        # Vertex AI SDK 방식으로 모델을 초기화합니다.
        # 모델 이름에 'publishers/google/models/' 부분이 필요 없습니다.
        gemini_model = GenerativeModel("gemini-2.5-pro", system_instruction=SYSTEM_PROMPT)
        
        storage_client = storage.Client(project=PROJECT_ID)

        clients_initialized = True
        print("Vertex AI SDK clients initialized successfully.")

    except Exception as e:
        print(f"CRITICAL: Failed to initialize clients. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

@app.route("/", methods=["POST"])
def process_upload_event():
    try:
        init_clients()

        envelope = request.json
        if not envelope or "message" not in envelope:
            return jsonify({"error": "Invalid Pub/Sub message format"}), 400

        pubsub_message = envelope["message"]
        if "data" not in pubsub_message:
            return jsonify({"error": "Pub/Sub message data missing"}), 400

        decoded_data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
        message_json = json.loads(decoded_data)
        bucket_name = message_json.get("bucket")
        file_name = message_json.get("name")
        if not bucket_name or not file_name:
            return jsonify({"error": "Missing 'bucket' or 'name' in message data"}), 400

        # ==================================================================
        # 1. 멱등성 확인 (Idempotency Check) - 핵심 변경 사항
        # ==================================================================
        # 결과물이 저장될 버킷과 파일 접두사를 정의합니다.
        # (예: 'folder/doc.pdf' -> 'folder/doc-page-')
        destination_bucket = storage_client.bucket(PARSED_BUCKET_NAME)
        prefix = f"{file_name.replace('.pdf', '')}-page-"

        # list_blobs를 사용하여 해당 prefix를 가진 파일이 하나라도 있는지 확인합니다.
        # max_results=1은 매우 효율적인 확인 방법입니다.
        existing_blobs = destination_bucket.list_blobs(prefix=prefix, max_results=1)
        if next(existing_blobs, None):
            print(f"File '{file_name}' has already been processed. Skipping.")
            # 204 No Content는 Pub/Sub에 성공적으로 처리했음을 알려 메시지 재전송을 막습니다.
            return "OK", 204
        # ==================================================================
        
        print(f"Processing file: gs://{bucket_name}/{file_name}")

        source_bucket = storage_client.bucket(bucket_name)
        pdf_blob = source_bucket.blob(file_name)
        pdf_bytes = pdf_blob.download_as_bytes()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        destination_bucket = storage_client.bucket(PARSED_BUCKET_NAME)

        for i, page in enumerate(pdf_document):
            page_num = i + 1
            page_text = page.get_text().strip()
            image_bytes = page.get_pixmap(dpi=150).tobytes("png")

            print(f"  - Processing page {page_num}...")

            # ==================================================================
            # 2. Crash 방지 로직
            # ==================================================================
            # 페이지에서 추출된 텍스트가 비어 있는지 확인합니다.
            if not page_text:
                # 텍스트가 비어있으면 Gemini 에러 방지를 위해 대체 텍스트를 사용합니다.
                page_text_for_gemini = "(No text content found on this page)"
                print(f"    - Page {page_num} has no text. Using placeholder.")
            else:
                page_text_for_gemini = page_text
            # ==================================================================
                        
            # Vertex AI SDK가 권장하는 'Part' 객체를 사용하여 데이터를 안전하게 전달합니다.
            image_part = Part.from_data(data=image_bytes, mime_type="image/png")
            
            # Gemini에 보낼 때는 항상 텍스트가 있도록 보장된 변수를 사용합니다.
            response = gemini_model.generate_content(
                [image_part, page_text_for_gemini],
                generation_config=GenerationConfig(response_mime_type="application/json")
            )
            
            metadata_json = json.loads(response.text)

            intermediate_data = {
                "source_file": f"gs://{bucket_name}/{file_name}",
                "page_num": page_num,
                "page_text": page_text,
                "image_base64": base64.b64encode(image_bytes).decode('utf-8'),
                "metadata": metadata_json
            }

            new_blob_name = f"{file_name.replace('.pdf', '')}-page-{page_num}.json"
            new_blob = destination_bucket.blob(new_blob_name)
            new_blob.upload_from_string(
                json.dumps(intermediate_data, ensure_ascii=False),
                content_type="application/json"
            )
            print(f"  - Saved intermediate data to gs://{PARSED_BUCKET_NAME}/{new_blob_name}")

        pdf_document.close()
        print(f"Successfully processed {file_name}")
        return "OK", 200

    except Exception as e:
        print(f"Error processing upload event: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500