### **Phase 1: 환경 설정 및 사전 준비 (Setup & Prerequisites)**

가장 먼저 아키텍처에 필요한 Google Cloud 서비스들을 활성화하고, 데이터베이스와 스토리지, 통신 채널을 설정해야 합니다.

#### **1. Google Cloud API 활성화**

Cloud Shell 또는 로컬 터미널에서 다음 명령어를 실행하여 필요한 모든 API를 활성화합니다.

```bash
gcloud services enable \
    aiplatform.googleapis.com \
    run.googleapis.com \
    pubsub.googleapis.com \
    sqladmin.googleapis.com \
    alloydb.googleapis.com \
    storage.googleapis.com \
    cloudbuild.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    bigquery.googleapis.com
```

#### **2. AlloyDB for PostgreSQL 설정**

벡터 데이터를 저장할 AlloyDB를 생성하고, 벡터 검색을 위한 `pgvector` 확장을 활성화합니다.

1.  **AlloyDB 클러스터 및 인스턴스 생성:** [Cloud Console](https://console.cloud.google.com/alloydb)을 통해 또는 `gcloud` 명령어로 클러스터와 기본 인스턴스를 생성합니다.
2.  **데이터베이스 생성:** 인스턴스에 연결하여 사용할 데이터베이스를 생성합니다. (예: `poc_db`)
3.  **pgvector 확장 활성화 및 테이블 생성:**
    `psql` 등을 통해 데이터베이스에 접속한 후, 다음 SQL 쿼리를 실행합니다.

    ```sql
    -- pgvector 확장 기능 활성화
    CREATE EXTENSION IF NOT EXISTS vector;

    -- 벡터 데이터와 원본 텍스트를 저장할 테이블 생성
    CREATE TABLE document_embeddings (
        id SERIAL PRIMARY KEY,
        source_file VARCHAR(1024),
        chunk_description TEXT, -- 이 청크가 무엇에 대한 것인지 설명 (예: "3페이지의 아키텍처 다이어그램")
        embedding VECTOR(1408) -- 멀티모달 임베딩 모델의 차원 수
    );

    -- 벡터 검색 속도를 높이기 위한 HNSW 인덱스 생성 (선택 사항이지만 강력히 권장)
    CREATE INDEX ON document_embeddings
    USING hnsw (embedding vector_l2_ops);
    ```

#### **3. Pub/Sub 토픽 생성**

GCS 파일 업로드 이벤트를 수신할 Pub/Sub 토픽을 생성합니다.

```bash
gcloud pubsub topics create gcs-file-events
```

#### **4. GCS 버킷 알림 설정**

샘플 데이터가 업로드된 GCS 버킷에 파일이 생성될 때마다 위에서 만든 Pub/Sub 토픽으로 알림을 보내도록 설정합니다.

```bash
gcloud storage buckets notifications create gs://[YOUR_GCS_BUCKET_NAME] \
    --topic=gcs-file-events \
    --event-types=OBJECT_FINALIZE
```

---

### **Phase 2: 데이터 수집 및 처리 서브시스템 (Data Ingestion & Processing)**

이 단계에서는 GCS에 파일이 업로드되면 이를 감지하여 내용을 분석/추출/청킹하고, 벡터로 변환하여 AlloyDB에 저장하는 Cloud Run 서비스를 구현합니다.

#### **1. Cloud Run 서비스 코드 작성 (File Processor)**

이 서비스는 Pub/Sub 메시지를 받아 해당 파일을 처리하는 역할을 합니다. Gemini 1.5 Pro의 멀티모달 기능을 활용하면 PDF, 이미지(JPG, PNG), Excel(XLS) 등 다양한 형식의 파일을 단일 모델로 처리할 수 있습니다.

**`main.py` (Python, Flask 사용 예시)**

```python
import base64
import json
import os
import re

import vertexai
from flask import Flask, request
from google.cloud import alloydb, storage
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part

# --- 환경 변수 및 클라이언트 초기화 ---
PROJECT_ID = os.environ.get("GCP_PROJECT")
REGION = os.environ.get("REGION", "asia-northeast3")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
DB_INSTANCE_IP = os.environ.get("DB_INSTANCE_IP") # AlloyDB Private IP

# Vertex AI 초기화
vertexai.init(project=PROJECT_ID, location=REGION)

# 스토리지 클라이언트
storage_client = storage.Client()

# AlloyDB 커넥터
db_connector = alloydb.Connector()

# Gemini 1.5 Pro 모델 로드
multimodal_model = GenerativeModel("gemini-1.5-pro-001")

# Flask 앱
app = Flask(__name__)

# --- Helper Functions ---

def get_db_connection():
    """AlloyDB와 안전하게 연결합니다."""
    conn = db_connector.connect(
        f"projects/{PROJECT_ID}/locations/{REGION}/clusters/[YOUR_ALLOYDB_CLUSTER_ID]/instances/[YOUR_ALLOYDB_INSTANCE_ID]",
        "pg8000",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
        ip_type="PRIVATE" # VPC 내에서 통신
    )
    return conn

def get_text_embedding(text):
    """주어진 텍스트에 대한 임베딩 벡터를 생성합니다."""
    model = GenerativeModel("text-embedding-004")
    result = model.generate_content([text])
    # 첫 번째 콘텐츠의 첫 번째 부분에서 임베딩 값을 추출
    if result.candidates and result.candidates[0].content.parts:
        return result.candidates[0].content.parts[0].embedding
    return None


def extract_and_chunk_content(bucket_name, file_name):
    """
    GCS에서 파일을 다운로드하고 Gemini를 사용하여 텍스트를 추출 및 청킹합니다.
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    file_bytes = blob.download_as_bytes()
    
    # MIME 타입 추론 (파일 확장자 기반)
    mime_type = blob.content_type
    if not mime_type:
        if file_name.lower().endswith('.pdf'):
            mime_type = 'application/pdf'
        elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            mime_type = f'image/{file_name.lower().split(".")[-1]}'
        # 기타 타입 추가 가능 (e.g., 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' for xlsx)
        # Gemini 1.5 Pro는 다양한 포맷을 지원합니다.
    
    # Gemini 1.5 Pro에 보낼 프롬프트 구성
    prompt = """
    당신은 문서 분석 전문가입니다. 주어진 파일(이미지, PDF, 오피스 문서 등)의 내용을 분석하여 가장 핵심적인 텍스트 정보를 추출하고, 의미적으로 완결된 여러 개의 단락(chunk)으로 나누어주세요. 각 단락은 검색 및 답변 생성에 사용하기 좋은 크기여야 합니다.

    출력 형식은 반드시 아래의 JSON 형식이어야 합니다. 다른 설명은 추가하지 마세요.

    {
      "chunks": [
        "첫 번째 텍스트 단락입니다.",
        "두 번째 의미 있는 텍스트 단락입니다.",
        "..."
      ]
    }
    """
    
    # 멀티모달 요청 생성
    file_part = Part.from_data(data=file_bytes, mime_type=mime_type)
    generation_config = GenerationConfig(response_mime_type="application/json")
    
    response = multimodal_model.generate_content([prompt, file_part], generation_config=generation_config)
    
    try:
        # JSON 응답 파싱
        response_json = json.loads(response.text)
        return response_json.get("chunks", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing Gemini response: {e}\nResponse text: {response.text}")
        return []

# --- Flask 라우트 ---

@app.route("/", methods=["POST"])
def process_pubsub_event():
    envelope = request.get_json()
    if not envelope or "message" not in envelope:
        print("Bad Request: invalid Pub/Sub message format")
        return "Bad Request", 400

    pubsub_message = envelope["message"]
    if "data" in pubsub_message:
        # 데이터는 base64로 인코딩되어 있습니다.
        data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
        message_json = json.loads(data)
        
        bucket_name = message_json.get("bucket")
        file_name = message_json.get("name")

        if not bucket_name or not file_name:
            print(f"Invalid message format: {message_json}")
            return "Bad Request", 400

        print(f"Processing file: gs://{bucket_name}/{file_name}")

        # 1. Gemini로 콘텐츠 추출 및 청킹
        chunks = extract_and_chunk_content(bucket_name, file_name)
        if not chunks:
            print("No content chunks extracted.")
            return "OK", 204

        # 2. 각 청크를 임베딩하고 DB에 저장
        conn = get_db_connection()
        with conn.cursor() as cursor:
            for chunk in chunks:
                if not chunk.strip(): # 비어있는 청크는 건너뛰기
                    continue
                
                # 임베딩 생성
                embedding = get_text_embedding(chunk)
                if embedding:
                    # AlloyDB에 저장
                    cursor.execute(
                        "INSERT INTO document_embeddings (source_file, content, embedding) VALUES (%s, %s, %s)",
                        (f"gs://{bucket_name}/{file_name}", chunk, embedding)
                    )
        conn.commit()
        conn.close()

        print(f"Successfully processed and embedded {len(chunks)} chunks from {file_name}.")
        return "OK", 204
    
    return "OK", 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

```

**`requirements.txt`**

```
Flask==3.0.0
gunicorn==21.2.0
google-cloud-aiplatform>=1.47.0
google-cloud-storage>=2.14.0
google-cloud-alloydb[pg8000]>=0.1.0
```

#### **2. Cloud Run 서비스 배포**

위에서 작성한 코드를 Cloud Run에 배포합니다.

1.  **VPC 커넥터 생성:** Cloud Run이 AlloyDB의 Private IP와 통신하려면 [서버리스 VPC 액세스 커넥터](https://console.cloud.google.com/vpc/connectors)가 필요합니다. AlloyDB가 있는 VPC 네트워크에 커넥터를 생성합니다.

2.  **서비스 배포:**
    `main.py`와 `requirements.txt`가 있는 디렉토리에서 다음 명령어를 실행합니다. `[...]` 부분을 실제 값으로 대체하세요.

    ```bash
    gcloud run deploy file-processor-service \
        --source . \
        --platform managed \
        --region [YOUR_REGION] \
        --allow-unauthenticated \
        --vpc-connector [YOUR_VPC_CONNECTOR_NAME] \
        --set-env-vars "REGION=[YOUR_REGION],DB_USER=[YOUR_DB_USER],DB_PASS=[YOUR_DB_PASSWORD],DB_NAME=[YOUR_DB_NAME],DB_INSTANCE_IP=[YOUR_ALLOYDB_INSTANCE_IP]" \
        --service-account [SERVICE_ACCOUNT_EMAIL] # 아래 참고
    ```

    *   **서비스 계정 권한:** 배포에 사용되는 서비스 계정(`[SERVICE_ACCOUNT_EMAIL]`)은 다음 역할을 가지고 있어야 합니다:
        *   Vertex AI 사용자 (roles/aiplatform.user)
        *   AlloyDB 클라이언트 (roles/alloydb.client)
        *   Storage 객체 뷰어 (roles/storage.objectViewer)

#### **3. Pub/Sub 구독 생성**

GCS 알림 토픽과 `file-processor-service`를 연결하는 Push 구독을 생성합니다.

```bash
gcloud pubsub subscriptions create gcs-file-event-subscription \
    --topic gcs-file-events \
    --push-endpoint=[YOUR_CLOUD_RUN_SERVICE_URL] \
    --push-auth-service-account=[CLOUD_RUN_INVOKER_SERVICE_ACCOUNT_EMAIL]
```

이제 GCS 버킷에 파일이 업로드되면 자동으로 Cloud Run 서비스가 실행되어 파일 내용을 분석, 임베딩 후 AlloyDB에 저장합니다.

---

### **Phase 3: 서빙 서브시스템 (Serving Subsystem)**

이제 AlloyDB에 저장된 벡터 데이터를 활용하여 사용자 질문에 답변하는 챗봇 에이전트를 구성합니다. 다이어그램의 `Google AgentSpace`는 **Vertex AI Agent Builder** (과거 Gen App Builder)를 의미하는 것으로 보입니다.

1.  **Vertex AI Agent Builder에서 앱 생성:**
    *   [Vertex AI Agent Builder 콘솔](https://console.cloud.google.com/gen-app-builder)로 이동합니다.
    *   새로운 `검색(Search)` 또는 `채팅(Chat)` 앱을 생성합니다.
2.  **데이터 저장소(Data Store) 구성:**
    *   앱 설정에서 '데이터 저장소' 섹션으로 이동하여 '새 데이터 저장소'를 만듭니다.
    *   데이터 소스로 **'AlloyDB'**를 선택합니다.
    *   DB 연결 정보를 입력합니다: 프로젝트 ID, 인스턴스 ID, 데이터베이스 이름, 테이블 이름(`document_embeddings`), 사용자 인증 정보 등.
    *   **콘텐츠 열**에는 `content`를, **임베딩 열**에는 `embedding`을 매핑합니다.
3.  **에이전트 테스트:**
    *   데이터 저장소 생성이 완료되면 Agent Builder의 미리보기(Preview) 기능에서 바로 질문을 입력하여 테스트할 수 있습니다.
    *   에이전트는 사용자의 질문을 자동으로 임베딩하고, AlloyDB에서 벡터 검색(Semantic Search)을 수행하여 가장 관련성 높은 `content`를 찾은 후, Gemini 모델을 사용하여 최종 답변을 생성합니다.

---

### **Phase 4: 품질 평가 서브시스템 (Quality Evaluation)**

이 시스템은 생성된 답변의 품질을 자동으로 평가하고 결과를 BigQuery에 저장하는 역할을 합니다.

#### **1. 평가용 Cloud Run Job 코드**

이 코드는 평가 데이터셋(질문-정답 쌍)을 기반으로 서빙 에이전트에 질문을 던지고, 그 답변을 평가 모델(예: Gemini)을 사용해 채점합니다.

**`evaluator_job.py` (예시)**

```python
# (필요한 라이브러리 import)
# ...

def get_evaluation_dataset():
    # 평가용 질문-정답 쌍을 가져옵니다. (예: GCS 파일, BigQuery 등)
    return [
        {"question": "이 문서의 주요 내용은 무엇인가요?", "golden_answer": "이 문서는 AI 아키텍처에 대한 설명입니다."},
        # ...
    ]

def query_serving_agent(question):
    # Vertex AI Agent Builder SDK 또는 API를 사용하여 질문을 보내고 답변을 받습니다.
    # 이 부분은 Agent Builder API 문서를 참고하여 구현해야 합니다.
    # response = agent.search(question)
    # return response.answer
    return "에이전트로부터 받은 답변 예시" # Placeholder

def evaluate_response(question, generated_answer, golden_answer):
    # Gemini를 평가 모델로 사용하여 답변 품질을 채점합니다.
    evaluator_model = GenerativeModel("gemini-1.5-pro-001")
    prompt = f"""
    Question: {question}
    Golden Answer: {golden_answer}
    Generated Answer: {generated_answer}

    'Generated Answer'가 'Golden Answer'와 얼마나 일치하고 질문에 대해 정확한지 1~5점 척도로 평가하고, 그 이유를 간략히 설명해주세요.
    반드시 아래 JSON 형식으로만 답변해주세요.

    {{
      "score": <1에서 5 사이의 정수>,
      "reason": "<평가 이유>"
    }}
    """
    response = evaluator_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

def save_to_bigquery(results):
    # 평가 결과를 BigQuery 테이블에 저장합니다.
    # bq_client = bigquery.Client()
    # bq_client.insert_rows_json(...)
    print("Saving results to BigQuery:", results) # Placeholder

def main():
    evaluation_set = get_evaluation_dataset()
    evaluation_results = []

    for item in evaluation_set:
        question = item["question"]
        golden_answer = item["golden_answer"]
        
        generated_answer = query_serving_agent(question)
        evaluation = evaluate_response(question, generated_answer, golden_answer)
        
        result_row = {
            "prompt": question,
            "response": generated_answer,
            "evaluation_score": evaluation["score"],
            "evaluation_reason": evaluation["reason"]
        }
        evaluation_results.append(result_row)

    save_to_bigquery(evaluation_results)

if __name__ == "__main__":
    main()
```

#### **2. BigQuery 테이블 생성**

평가 결과를 저장할 테이블을 미리 생성합니다.

```sql
CREATE TABLE `[YOUR_PROJECT_ID].[YOUR_DATASET_ID].evaluation_results` (
    prompt STRING,
    response STRING,
    evaluation_score INT64,
    evaluation_reason STRING,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
```

#### **3. 평가 실행**

*   **Cloud Run Job으로 배포:** 위 코드를 Job 형태로 배포하여 필요할 때마다 실행할 수 있습니다.
*   **Trigger 설정:** Cloud Scheduler와 Pub/Sub을 사용하여 정기적으로(예: 매일 밤) 평가 작업을 트리거하도록 구성할 수 있습니다.

---

이 가이드가 제시해주신 아키텍처를 성공적으로 구현하는 데 도움이 되기를 바랍니다. 각 단계별로 필요한 권한이나 네트워크 설정 등 세부적인 부분에서 문제가 발생하면 언제든지 추가 질문을 남겨주세요.