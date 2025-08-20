import base64
import json
import os
import ssl
import sys
import traceback

from flask import Flask, request
import pg8000.dbapi
from google.cloud import storage

# readme.md 가이드에 따라 Vertex AI SDK를 사용합니다.
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as VisionImage, MultiModalEmbeddingModel

# --- 환경 변수 및 전역 변수 설정 ---
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
PROJECT_ID = os.environ.get("GCP_PROJECT")
EMBEDDING_REGION = os.environ.get("EMBEDDING_REGION", "asia-northeast3")

# 클라이언트 초기화 상태를 관리하는 전역 변수
clients_initialized = False
text_embedding_model = None
multimodal_embedding_model = None
storage_client = None

app = Flask(__name__)

def init_clients():
    """Vertex AI 및 GCS 클라이언트를 초기화합니다."""
    global clients_initialized, text_embedding_model, multimodal_embedding_model, storage_client
    if clients_initialized:
        return

    print(f"Initializing clients with Vertex AI SDK for project '{PROJECT_ID}' in '{EMBEDDING_REGION}'...")
    try:
        # Vertex AI SDK는 GCP 환경의 컨텍스트(프로젝트, 인증)를 자동으로 인식합니다.
        # location은 사용할 모델이 있는 리전을 명시합니다.
        vertexai.init(project=PROJECT_ID, location=EMBEDDING_REGION)

        # 모델과 스토리지 클라이언트를 초기화합니다.
        text_embedding_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
        multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        storage_client = storage.Client(project=PROJECT_ID)

        clients_initialized = True
        print("Embedder clients initialized successfully.")

    except Exception as e:
        print(f"CRITICAL: Failed to initialize clients. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

def get_db_connection():
    """AlloyDB 데이터베이스에 대한 보안 연결을 생성합니다."""
    print("Connecting to AlloyDB...")
    try:
        # 기본 컨텍스트 대신, 서버 인증서 검증을 수행하지 않는 커스텀 컨텍스트를 생성합니다.
        # 이는 VPC 내부의 안전한 통신 환경을 전제로 합니다.
        ssl_context = ssl.SSLContext()
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl_context.check_hostname = False
        
        conn = pg8000.dbapi.connect(
            host=DB_HOST, port=5432, user=DB_USER, password=DB_PASS, database=DB_NAME, ssl_context=ssl_context
        )
        print("AlloyDB connection successful.")
        return conn
    except Exception as e:
        print(f"CRITICAL: Failed to connect to AlloyDB. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

@app.route("/", methods=["POST"])
def process_parsed_event():
    """Pub/Sub으로부터 받은 이벤트를 처리하여 임베딩을 생성하고 DB에 저장합니다."""
    envelope = request.get_json()
    if not envelope or "message" not in envelope:
        print("ERROR: Bad Pub/Sub request format.", file=sys.stderr)
        return "Bad Request: Invalid Pub/Sub message format", 400

    # Base64로 인코딩된 실제 메시지(GCS 이벤트 알림)를 디코딩합니다.
    pubsub_message = base64.b64decode(envelope["message"]["data"]).decode("utf-8")
    message_json = json.loads(pubsub_message)

    bucket_name = message_json.get("bucket")
    file_name = message_json.get("name")

    if not bucket_name or not file_name:
        print(f"ERROR: Invalid GCS event data: {message_json}", file=sys.stderr)
        return "Bad Request: Invalid GCS event data", 400

    try:
        # 1. 클라이언트 초기화 (최초 실행 시 1회)
        init_clients()

        # 2. GCS에서 파서 서비스가 생성한 중간 JSON 파일 다운로드
        print(f"Processing gs://{bucket_name}/{file_name}...")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        intermediate_data = json.loads(blob.download_as_string())

        # 3. 임베딩 생성을 위한 데이터 추출
        summary = intermediate_data["metadata"]["summary"]
        page_text = intermediate_data["page_text"]
        image_bytes = base64.b64decode(intermediate_data["image_base64"])

        # 4. 텍스트 및 멀티모달 임베딩 생성
        print("  - Generating text embedding for summary...")
        text_embeddings = text_embedding_model.get_embeddings([summary])

        # ==================================================================
        # 멀티모달 임베딩 시 텍스트를 제외하고 이미지만 사용
        # ==================================================================
        # Multi-modal 모델의 contextual_text는 1024자 제한이 있습니다.
        #if len(page_text) > 1024:
        #    print(f"    - page_text is too long ({len(page_text)} chars). Truncating to 1024.")
        #    contextual_text_for_embedding = page_text[:1024]
        #else:
        #    contextual_text_for_embedding = page_text
        #
        #print("  - Generating multimodal embedding for image and text...")
        #multimodal_embeddings = multimodal_embedding_model.get_embeddings(
        #    image=VisionImage(image_bytes=image_bytes),
        #    contextual_text=page_text
        #)
        # ==================================================================
        print("  - Generating multimodal embedding for image only...")
        # contextual_text 파라미터를 완전히 제거하여 순수 이미지 임베딩을 생성합니다.
        multimodal_embeddings = multimodal_embedding_model.get_embeddings(
            image=VisionImage(image_bytes=image_bytes)
        )

        # 5. 최종 결과를 AlloyDB에 저장
        insert_data = (
            intermediate_data["source_file"],
            f"Content from page {intermediate_data['page_num']} of file {os.path.basename(intermediate_data['source_file'])}",
            json.dumps(intermediate_data["metadata"], ensure_ascii=False),
            # 벡터 리스트를 pgvector가 이해하는 JSON 문자열로 변환
            json.dumps(text_embeddings[0].values),
            json.dumps(multimodal_embeddings.image_embedding)
        )

        # ==================================================================
        # 핵심 변경 사항: 들여쓰기 오류 해결 및 DB 핸들링 구조 개선
        # ==================================================================
        print(f"  - Inserting record for {intermediate_data['source_file']} page {intermediate_data['page_num']} into AlloyDB...")
        conn = None
        cursor = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO document_embeddings
                   (source_file, chunk_description, metadata, text_embedding, multimodal_embedding)
                   VALUES (%s, %s, %s, %s::vector, %s::vector)""",
                insert_data
            )
            conn.commit()
        finally:
            # 커서와 연결이 성공적으로 생성되었을 경우에만 닫기를 시도
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        # ==================================================================

        print(f"Successfully processed and inserted data for gs://{bucket_name}/{file_name}.")
        return "OK", 204

    except Exception as e:
        print(f"CRITICAL: Failed to process file {file_name}. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        # Pub/Sub이 재시도할 수 있도록 에러 응답을 반환합니다.
        return "Internal Server Error", 500

if __name__ == "__main__":
    # Gunicorn을 통해 실행되므로, 이 부분은 로컬 테스트 용도로만 사용됩니다.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))