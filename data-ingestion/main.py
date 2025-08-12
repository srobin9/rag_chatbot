import base64
import io
import json
import os
import ssl
import traceback
import sys # 상세한 traceback 로깅을 위해 추가

import pandas as pd
import pg8000.dbapi
import vertexai
from flask import Flask, request
from google.cloud import storage
import fitz  # PyMuPDF

# --- Vertex AI SDK 및 모델 클래스 임포트 (UPDATED) ---
from vertexai.generative_models import GenerativeModel, Part, Tool
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as VisionImage
from vertexai.vision_models import MultiModalEmbeddingModel

# --- 환경 변수 및 클라이언트 초기화 ---
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
PROJECT_ID = os.environ.get("GCP_PROJECT")
REGION = os.environ.get("REGION", "asia-northeast3")

# --- 전역 클라이언트 변수 ---
vertexai_initialized = False
gemini_model = None
text_embedding_model = None
multimodal_embedding_model = None
storage_client = None
grounding_tool = None 

# AlloyDB SSL 컨텍스트
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# --- 프롬프트 정의 ---
# Grounding을 활용하도록 프롬프트를 좀 더 구체화할 수 있습니다.
GEMINI_PROMPT = """
당신은 문서를 분석하여 구조화된 JSON으로 만드는 전문가입니다.
주어진 문서 청크(이미지와 텍스트)를 분석하고, 필요하다면 웹 검색을 통해 최신 정보를 확인하여 아래 지침에 따라 JSON 형식으로 결과를 반환해 주세요.

1.  **summary**: 문서 청크의 핵심 내용을 한국어로 명확하고 간결하게 요약합니다. 문서에 언급된 특정 용어, 인물, 회사에 대한 최신 정보가 필요하면 웹을 참조하여 요약에 반영하세요.
2.  **extracted_entities**: 문서에서 발견된 중요한 개체(날짜, 사람, 회사, 금액, 기술 용어 등)를 추출합니다.
3.  **document_type**: 이 청크를 기반으로 문서의 유형(예: 계약서, 최신 기술 보고서, 금융 분석, 뉴스 기사, 다이어그램)을 추정합니다.
4.  **keywords**: 내용의 핵심 키워드를 5개 이내로 추출합니다.

모든 응답은 다른 설명 없이 오직 JSON 객체만 포함해야 합니다.
"""

app = Flask(__name__)

# --- 클라이언트 초기화를 담당하는 함수 ---
def init_clients():
    # *** 중요: grounding_tool을 global 목록에 추가 ***
    global vertexai_initialized, gemini_model, text_embedding_model, multimodal_embedding_model, storage_client, grounding_tool
    
    if vertexai_initialized:
        return

    print("Attempting to initialize clients...")
    try:
        print(f"Initializing Vertex AI for project '{PROJECT_ID}' in region '{REGION}'")
        vertexai.init(project=PROJECT_ID, location=REGION)
        
        print("Loading GenerativeModel 'gemini-2.5-pro'...")
        gemini_model = GenerativeModel("gemini-2.5-pro")

        print("Loading TextEmbeddingModel 'text-multilingual-embedding-002'...")
        text_embedding_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

        print("Loading MultiModalEmbeddingModel 'multimodalembedding@001'...")
        multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        
        print("Creating Google Search grounding tool...")
        grounding_tool = Tool.from_google_search_retrieval()

        print("Initializing Google Cloud Storage client...")
        storage_client = storage.Client()
       
        vertexai_initialized = True
        print("All clients initialized successfully.")
    except Exception as e:
        # *** 중요: 초기화 실패 시 정확한 에러를 로깅하고 다시 예외를 발생시켜 앱 비정상 상태 방지 ***
        print(f"CRITICAL: Failed to initialize clients. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

# --- Helper Functions ---
def get_db_connection():
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

# --- Processing Functions (UPDATED) ---
def process_pdf(blob, conn):
    """Grounding을 포함한 하이브리드 전략으로 PDF를 처리"""
    pdf_bytes = blob.download_as_bytes()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    with conn.cursor() as cursor:
        for i, page in enumerate(pdf_document):
            page_text = page.get_text().strip()
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            
            # --- 1. Gemini 2.5 Pro로 메타데이터 추출 (Grounding 사용) ---
            response = gemini_model.generate_content(
                [GEMINI_PROMPT, Part.from_data(image_bytes, mime_type="image/png"), page_text],
                tools=[grounding_tool]
            )
            metadata_json = get_json_from_gemini_response(response)
            
            if not metadata_json or not metadata_json.get("summary"):
                print(f"Skipping page {i+1} due to metadata/summary extraction failure.")
                continue

            summary = metadata_json["summary"]

            # --- 2. 텍스트 임베딩 생성 ---
            # task_type을 명시하여 검색 품질 향상
            text_embeddings = text_embedding_model.get_embeddings(
                [summary], task_type="RETRIEVAL_DOCUMENT"
            )
            text_vector = text_embeddings[0].values

            # --- 3. 멀티모달 임베딩 생성 ---
            vision_image = VisionImage(image_bytes=image_bytes)
            multi_embeddings = multimodal_embedding_model.embed_image(image=vision_image, contextual_text=page_text)
            multimodal_vector = multi_embeddings[0].values
            
            # --- 4. DB에 저장 ---
            description = f"Content from page {i+1} of file {blob.name}"
            cursor.execute(
                """INSERT INTO document_embeddings 
                   (source_file, chunk_description, metadata, text_embedding, multimodal_embedding) 
                   VALUES (%s, %s, %s, %s, %s)""",
                (f"gs://{blob.bucket.name}/{blob.name}", description, json.dumps(metadata_json, ensure_ascii=False), text_vector, multimodal_vector)
            )
    print(f"Processed {pdf_document.page_count} pages from {blob.name} with Grounding.")


def process_image(blob, conn):
    """Grounding을 포함한 하이브리드 전략으로 이미지 처리"""
    image_bytes = blob.download_as_bytes()
    
    with conn.cursor() as cursor:
        # --- 1. Gemini 2.5 Pro로 메타데이터 추출 (Grounding 사용) ---
        response = gemini_model.generate_content(
            [GEMINI_PROMPT, Part.from_data(image_bytes, mime_type="image/png")],
            tools=[grounding_tool]
        )
        metadata_json = get_json_from_gemini_response(response)
        
        if not metadata_json or not metadata_json.get("summary"):
            print(f"Skipping image {blob.name} due to metadata/summary extraction failure.")
            return

        summary = metadata_json["summary"]

        # --- 2. 텍스트 임베딩 생성 ---
        text_embeddings = text_embedding_model.get_embeddings([summary], task_type="RETRIEVAL_DOCUMENT")
        text_vector = text_embeddings[0].values
        
        # --- 3. 멀티모달 임베딩 생성 ---
        vision_image = VisionImage(image_bytes=image_bytes)
        multi_embeddings = multimodal_embedding_model.embed_image(image=vision_image)
        multimodal_vector = multi_embeddings[0].values
        
        # --- 4. DB에 저장 ---
        description = f"Image file: {blob.name}"
        cursor.execute(
            """INSERT INTO document_embeddings 
               (source_file, chunk_description, metadata, text_embedding, multimodal_embedding) 
               VALUES (%s, %s, %s, %s, %s)""",
            (f"gs://{blob.bucket.name}/{blob.name}", description, json.dumps(metadata_json, ensure_ascii=False), text_vector, multimodal_vector)
        )
    print(f"Processed image {blob.name} with Grounding.")


def process_excel(blob, conn):
    """Grounding을 포함한 하이브리드 전략으로 Excel 처리"""
    excel_bytes = blob.download_as_bytes()
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    
    with conn.cursor() as cursor:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None).fillna('')
            for i in range(0, len(df), 20): # 20행씩 청킹
                chunk_df = df[i:i+20]
                chunk_text = chunk_df.to_string()
                if not chunk_text.strip(): continue

                # --- 1. Gemini 2.5 Pro로 메타데이터 추출 (Grounding 사용) ---
                response = gemini_model.generate_content(
                    [GEMINI_PROMPT, chunk_text], tools=[grounding_tool]
                )
                metadata_json = get_json_from_gemini_response(response)

                if not metadata_json or not metadata_json.get("summary"):
                    print(f"Skipping chunk from sheet {sheet_name}, rows {i+1}-{i+20} due to metadata failure.")
                    continue

                summary = metadata_json["summary"]

                # --- 2. 텍스트 임베딩 생성 ---
                text_embeddings = text_embedding_model.get_embeddings([summary], task_type="RETRIEVAL_DOCUMENT")
                text_vector = text_embeddings[0].values

                # --- 3. 멀티모달 임베딩 생성 (텍스트만 사용) ---
                multi_embeddings = multimodal_embedding_model.embed_image(contextual_text=chunk_text)
                multimodal_vector = multi_embeddings[0].values
                
                # --- 4. DB에 저장 ---
                description = f"Content from sheet '{sheet_name}', rows {i+1}-{i+20} in file {blob.name}"
                cursor.execute(
                    """INSERT INTO document_embeddings 
                       (source_file, chunk_description, metadata, text_embedding, multimodal_embedding) 
                       VALUES (%s, %s, %s, %s, %s)""",
                    (f"gs://{blob.bucket.name}/{blob.name}", description, json.dumps(metadata_json, ensure_ascii=False), text_vector, multimodal_vector)
                )
    print(f"Processed sheets {xls.sheet_names} from {blob.name} with Grounding.")

# --- Flask 라우트 (상세 로깅 및 에러 핸들링 강화) ---
@app.route("/", methods=["POST"])
def process_pubsub_event():
    try:
        # --- 1. 클라이언트 초기화 ---
        # 이 단계에서 실패하면 서버 설정(권한, API 활성화 등) 문제임
        init_clients()
        
        # --- 2. Pub/Sub 메시지 파싱 ---
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
        # 초기화 또는 메시지 파싱 단계의 에러는 서버나 메시지 자체의 문제
        print(f"CRITICAL: Error during initialization or message parsing. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        # 이 단계의 에러는 재시도해도 성공할 가능성이 낮으므로 400 Bad Request 반환
        return "Bad Request", 400

    # --- 3. 파일 처리 로직 ---
    # 이 단계부터는 특정 파일에 대한 처리 로직
    conn = None
    try:
        print(f"Processing file: gs://{bucket_name}/{file_name}")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        conn = get_db_connection()
        
        file_ext = file_name.lower().split('.')[-1]

        if file_ext == 'pdf':
            process_pdf(blob, conn)
        elif file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
            process_image(blob, conn)
        elif file_ext in ['xls', 'xlsx']:
            process_excel(blob, conn)
        else:
            print(f"Unsupported file type: {file_ext}. Acknowledging message.")
            return "OK", 204 # 지원하지 않는 파일은 처리 성공으로 간주하여 재시도 방지
            
        conn.commit()
        print(f"Successfully processed and embedded file: {file_name}")
        return "OK", 204 # 성공

    except Exception as e:
        # 특정 파일을 처리하다가 발생한 에러
        print(f"ERROR: Failed to process file gs://{bucket_name}/{file_name}. Error: {e}", file=sys.stderr)
        traceback.print_exc()
        if conn:
            conn.rollback()
        
        # "Poison pill" 메시지의 무한 재시도를 막기 위해 200 OK를 반환.
        # Pub/Sub은 2xx 응답을 성공으로 간주하고 메시지를 다시 보내지 않음.
        # 만약 일시적인 오류일 수 있어 재시도를 원한다면 500 Internal Server Error를 반환해야 함.
        return "Error processed, preventing retry.", 200

    finally:
        if conn:
            conn.close()
            print("AlloyDB connection closed.")

if __name__ == "__main__":
    init_clients() # 로컬 실행 시 바로 초기화
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))