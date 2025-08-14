import base64, json, os, ssl, sys, traceback
from flask import Flask, request
import pg8000.dbapi
from google.cloud import storage
from google.api_core import client_options
from google.cloud.aiplatform.language_models import TextEmbeddingModel
from google.cloud.aiplatform.vision_models import Image as VisionImage, MultiModalEmbeddingModel

# --- Environment Variables & Global Placeholders ---
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
PROJECT_ID = os.environ.get("GCP_PROJECT")
EMBEDDING_REGION = os.environ.get("EMBEDDING_REGION", "asia-northeast3") # 임베딩 리전

# --- Global Client Variables (populated by init_clients) ---
clients_initialized = False
text_embedding_model = None
multimodal_embedding_model = None
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
    global clients_initialized, gemini_model, text_embedding_model, multimodal_embedding_model, storage_client
    if clients_initialized: return

    print(f"Attempting to initialize clients in project '{PROJECT_ID}'...")
    try:
        # 1. 새로운 google-generativeai SDK 설정 (Vertex AI 백엔드 사용)
        genai.configure(project=PROJECT_ID, location=MODEL_REGION)
        
        print(f"Loading GenerativeModel 'gemini-2.5-pro' via GenAI SDK from {MODEL_REGION}...")
        # 모델 이름 앞에 "publishers/google/models/"를 붙여 Vertex AI 모델임을 명시
        gemini_model = genai.GenerativeModel("publishers/google/models/gemini-2.5-pro")

        # 2. 기존 google-cloud-aiplatform SDK로 임베딩 모델 초기화
        # 각 서비스에 맞는 API 엔드포인트를 ClientOptions로 명확히 정의
        embedding_client_options = client_options.ClientOptions(api_endpoint=f"{EMBEDDING_REGION}-aiplatform.googleapis.com")

        print(f"Loading text embedding model 'text-multilingual-embedding-002' from {EMBEDDING_REGION}...")
        text_embedding_model = TextEmbeddingModel.from_pretrained(
            "text-multilingual-embedding-002",
            client_options=embedding_client_options
        )
        
        print(f"Loading multimodal embedding model 'multimodalembedding@001' from {EMBEDDING_REGION}...")
        multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(
            "multimodalembedding@001",
            client_options=embedding_client_options
        )

        print("Initializing Google Cloud Storage client...")
        storage_client = storage.Client(project=PROJECT_ID)

        clients_initialized = True
        print("All clients initialized successfully.")

    except Exception as e:
        print(f"CRITICAL: Failed to initialize clients. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

def get_db_connection():
    # ... (implementation from previous steps is fine)
    print("Connecting to AlloyDB...")
    try:
        # 보안 강화: pg8000에서 SSL 연결을 사용하려면 ssl_context가 필요합니다.
        # ssl.create_default_context()는 시스템의 기본 CA를 사용하여 인증서를 검증하므로
        # 보안 검증을 비활성화하는 것보다 훨씬 안전합니다.
        ssl_context = ssl.create_default_context()
        conn = pg8000.dbapi.connect(
            host=DB_HOST, port=5432, user=DB_USER, password=DB_PASS, database=DB_NAME, ssl_context=ssl_context
        )
        print("AlloyDB connection successful.")
        return conn
    except Exception as e:
        print(f"CRITICAL: Failed to connect to AlloyDB. Error: {e}", file=sys.stderr); traceback.print_exc(); raise

def get_json_from_gemini_response(response):
    # ... (implementation from previous steps is fine)
    try:
        # generation_config를 통해 response.text가 이미 유효한 JSON 문자열이어야 합니다.
        return json.loads(response.text)
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        print(f"ERROR: Could not parse JSON from Gemini response. Error: {e}\nResponse Text: {getattr(response, 'text', 'N/A')}", file=sys.stderr)
        return None

def process_pdf(blob, conn):
    pdf_bytes = blob.download_as_bytes()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages_data = []
    print(f"  - Preparing {pdf_document.page_count} pages for analysis from {blob.name}...")
    for i, page in enumerate(pdf_document):
        pages_data.append({
            "page_num": i + 1,
            "page_text": page.get_text().strip(),
            "image_bytes": page.get_pixmap(dpi=150).tobytes("png")
        })

    # --- 병렬 처리를 통한 메타데이터 추출 단계 ---
    pages_to_process = []
    # Gunicorn 스레드 수와 맞추거나 적절히 조절 (e.g., 8)
    with ThreadPoolExecutor(max_workers=8) as executor:
        print(f"  - Submitting {len(pages_data)} pages to Gemini for parallel processing...")

        future_to_page = {
            executor.submit(
                gemini_model.generate_content,
                # 새로운 SDK는 Part 객체 대신 단순 딕셔너리 구조를 사용
                [{"mime_type": "image/png", "data": p["image_bytes"]}, p["page_text"]],
                generation_config=genai.GenerationConfig(response_mime_type="application/json")
            ): p for p in pages_data
        }

        for future in as_completed(future_to_page):
            page_data = future_to_page[future]
            try:
                response = future.result()
                metadata_json = get_json_from_gemini_response(response)

                if not metadata_json or not metadata_json.get("summary"):
                    print(f"    - Skipping page {page_data['page_num']} due to missing summary or invalid JSON.")
                    continue

                page_data["summary"] = metadata_json["summary"]
                page_data["metadata_str"] = json.dumps(metadata_json, ensure_ascii=False)
                pages_to_process.append(page_data)

            except Exception as exc:
                print(f"    - Page {page_data['page_num']} generated an exception: {exc}")

    # Process pages in original order for deterministic batching
    pages_to_process.sort(key=lambda p: p["page_num"])

    if not pages_to_process:
        print(f"No valid pages found to process in {blob.name}.")
        return

    # --- API 배치 호출 단계 ---
    print(f"  - Batch embedding {len(pages_to_process)} text summaries...")
    summaries = [p["summary"] for p in pages_to_process]
    text_embeddings = text_embedding_model.get_embeddings(summaries)

    print(f"  - Batch embedding {len(pages_to_process)} multimodal inputs...")
    multimodal_embeddings = multimodal_embedding_model.embed_images(
        images=[VisionImage.from_bytes(p["image_bytes"]) for p in pages_to_process],
        contextual_texts=[p["page_text"] for p in pages_to_process]
    )

    # --- DB 배치 삽입 단계 ---
    insert_data = [
        (
            f"gs://{blob.bucket.name}/{blob.name}",
            f"Content from page {p['page_num']} of file {blob.name}",
            p["metadata_str"],
            text_embeddings[i].values,
            multimodal_embeddings[i].values
        ) for i, p in enumerate(pages_to_process)
    ]

    print(f"  - Inserting {len(insert_data)} records into the database...")
    cursor = conn.cursor()
    try:
        cursor.executemany("""INSERT INTO document_embeddings (source_file, chunk_description, metadata, text_embedding, multimodal_embedding) VALUES (%s, %s, %s, %s, %s)""", insert_data)
        conn.commit()
    finally:
        cursor.close()
    print(f"Successfully processed and inserted {len(pages_to_process)} pages from {blob.name}.")

@app.route("/", methods=["POST"])
def process_parsed_event():
    # ... Pub/Sub 메시지 파싱 ...
    bucket_name = message_json.get("bucket") # 여기서는 PARSED_BUCKET_NAME
    file_name = message_json.get("name")    # 여기서는 JSON 파일 이름

    # 중간 데이터 JSON 다운로드
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    intermediate_data = json.loads(blob.download_as_string())

    summary = intermediate_data["metadata"]["summary"]
    page_text = intermediate_data["page_text"]
    image_bytes = base64.b64decode(intermediate_data["image_base64"])
    
    # 텍스트 임베딩 생성
    text_embeddings = text_embedding_model.get_embeddings([summary])

    # 멀티모달 임베딩 생성
    multimodal_embeddings = multimodal_embedding_model.embed_images(
        images=[VisionImage.from_bytes(image_bytes)],
        contextual_texts=[page_text]
    )

    # DB에 최종 결과 저장
    insert_data = [(
        intermediate_data["source_file"],
        f"Content from page {intermediate_data['page_num']} of file {os.path.basename(intermediate_data['source_file'])}",
        json.dumps(intermediate_data["metadata"], ensure_ascii=False),
        text_embeddings[0].values,
        multimodal_embeddings[0].values
    )]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    # ... cursor.executemany(...) 및 DB 로직 ...
    conn.commit()
    conn.close()

    return "OK", 204