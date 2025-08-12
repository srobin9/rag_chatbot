import base64
import io
import json
import os
import pg8000.dbapi
import ssl

import pandas as pd
import vertexai
from flask import Flask, request
from google.cloud import storage
from smart_open import open # smart_open 라이브러리 사용 추천
from PIL import Image
from pypdf import PdfReader
from vertexai.vision_models import Image as VisionImage
from vertexai.vision_models import MultiModalEmbeddingModel

# --- 환경 변수 및 클라이언트 초기화 ---
PROJECT_ID = os.environ.get("GCP_PROJECT")
REGION = os.environ.get("REGION", "asia-northeast3")
DB_HOST = os.environ.get("DB_HOST")  # PSC DNS 이름
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")

# Vertex AI 초기화
vertexai.init(project=PROJECT_ID, location=REGION)

# 클라이언트 초기화
storage_client = storage.Client()
embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

app = Flask(__name__)

# AlloyDB는 암호화된 연결(SSL)이 필요합니다.
# Cloud Run 환경에 맞게 SSL 컨텍스트를 생성합니다.
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# --- Helper Functions ---
def get_db_connection():
    """VPC 커넥터를 통해 PSC 엔드포인트에 직접 연결합니다."""
    try:
        conn = pg8000.dbapi.connect(
            host=DB_HOST,
            port=5432,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            ssl_context=ssl_context
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def embed_content(image_bytes=None, text=None):
    """
    텍스트, 이미지, 또는 둘 다를 사용하여 일관된 1408차원 벡터를 생성합니다.
    """
    vision_image = VisionImage(image_bytes=image_bytes) if image_bytes else None
    
    # 모델 호출
    # 텍스트만 있는 경우, contextual_text에만 값이 전달되어 텍스트 임베딩 역할을 수행합니다.
    # 이미지만 있는 경우, image에만 값이 전달됩니다.
    # 둘 다 있는 경우, 두 문맥을 모두 이해하여 벡터를 생성합니다.
    embeddings = embedding_model.embed_image(
        image=vision_image,
        contextual_text=text,
    )
    return embeddings[0].values


def process_pdf(blob, conn):
    """PDF 파일의 각 페이지를 이미지와 텍스트로 임베딩합니다."""
    pdf_bytes = blob.download_as_bytes()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    
    with conn.cursor() as cursor:
        for i, page in enumerate(reader.pages):
            # 페이지를 이미지로 렌더링 (Pillow 사용)
            # pypdf는 직접 렌더링 기능이 없으므로, PyMuPDF(fitz)와 같은 라이브러리 사용을 권장.
            # 여기서는 개념을 설명하기 위해 Pillow를 사용한 가상 코드를 작성합니다.
            # 실제 구현 시에는 PDF 페이지를 이미지로 변환하는 안정적인 라이브러리를 사용해야 합니다.
            # 예시: page.to_image() 와 같은 가상 함수
            # 우선은 텍스트만 추출하여 임베딩하는 것으로 대체합니다.
            # TODO: PDF 페이지를 이미지로 변환하는 로직 추가 (e.g., using PyMuPDF)
            
            page_text = page.extract_text()
            if not page_text.strip():
                continue

            # 텍스트만으로 임베딩 (이미지 변환 로직 추가 시 image_bytes도 전달)
            embedding = embed_content(text=page_text)
            
            description = f"Content from page {i+1} of file {blob.name}"
            cursor.execute(
                "INSERT INTO document_embeddings (source_file, chunk_description, embedding) VALUES (%s, %s, %s)",
                (f"gs://{blob.bucket.name}/{blob.name}", description, embedding)
            )

def process_image(blob, conn):
    """이미지 파일을 임베딩합니다."""
    image_bytes = blob.download_as_bytes()
    embedding = embed_content(image_bytes=image_bytes)
    
    with conn.cursor() as cursor:
        description = f"Image file: {blob.name}"
        cursor.execute(
            "INSERT INTO document_embeddings (source_file, chunk_description, embedding) VALUES (%s, %s, %s)",
            (f"gs://{blob.bucket.name}/{blob.name}", description, embedding)
        )

def process_excel(blob, conn):
    """Excel 파일의 각 시트를 텍스트로 변환하여 임베딩합니다."""
    excel_bytes = blob.download_as_bytes()
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    
    with conn.cursor() as cursor:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            sheet_text = df.to_string()
            
            embedding = embed_content(text=sheet_text)
            
            description = f"Content from sheet '{sheet_name}' in file {blob.name}"
            cursor.execute(
                "INSERT INTO document_embeddings (source_file, chunk_description, embedding) VALUES (%s, %s, %s)",
                (f"gs://{blob.bucket.name}/{blob.name}", description, embedding)
            )
            
# --- Flask 라우트 ---
@app.route("/", methods=["POST"])
def process_pubsub_event():
    # ... (기존과 동일한 Pub/Sub 메시지 파싱 로직) ...
    envelope = request.get_json()
    if not envelope or "message" not in envelope:
        return "Bad Request", 400

    pubsub_message = envelope["message"]
    data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
    message_json = json.loads(data)
    
    bucket_name = message_json.get("bucket")
    file_name = message_json.get("name")
    
    print(f"Processing file: gs://{bucket_name}/{file_name}")

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        conn = get_db_connection()

        file_ext = file_name.lower().split('.')[-1]

        if file_ext == 'pdf':
            process_pdf(blob, conn)
        elif file_ext in ['png', 'jpg', 'jpeg']:
            process_image(blob, conn)
        elif file_ext in ['xls', 'xlsx']:
            process_excel(blob, conn)
        else:
            print(f"Unsupported file type: {file_ext}")
            return "OK", 204
            
        conn.commit()
        conn.close()
        print(f"Successfully processed and embedded file: {file_name}")

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        # 오류 발생 시에도 Pub/Sub 재시도를 막기 위해 200번대 응답 반환
        return "Error", 200

    return "OK", 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))