### **RAG PoC 아키텍처 구축 가이드**

본 문서는 Google Cloud Platform(GCP) 상에 RAG(Retrieval-Augmented Generation) 시스템 PoC를 구축하는 전체 과정을 안내합니다. 아키텍처는 데이터 수집, 처리, 임베딩, 서빙 및 평가 파이프라인으로 구성됩니다.

---

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
    bigquery.googleapis.com \
    dns.googleapis.com
```

#### **2. AlloyDB for PostgreSQL 설정**

벡터 데이터를 저장할 AlloyDB를 생성하고, 벡터 검색을 위한 `pgvector` 확장을 활성화합니다.

1.  **AlloyDB 클러스터 및 인스턴스 생성:**
    *   Cloud Console 또는 `gcloud`를 사용하여 클러스터와 기본 인스턴스를 생성합니다. 이때, **Private Service Connect(PSC)를 사용**하여 비공개 IP 연결을 설정합니다.

2.  **PSC 연결 수락:**
    *   AlloyDB 인스턴스의 '연결' 탭으로 이동합니다.
    *   'PSC 엔드포인트' 섹션에서 상태가 **`⚠️ 확인 필요`**인 연결 요청을 **수락(Accept)**합니다.
    *   **[중요]** 만약 조직 정책 오류로 수락이 불가능하다면, [트러블슈팅](#q-alloydb-psc-엔드포인트-연결-수락이-실패합니다) 섹션을 참조하세요.
    *   수락이 완료되면 엔드포인트 상태가 **`✅ 수락됨`**으로 바뀌고, 내부 IP와 **DNS 이름**이 할당됩니다. 이 DNS 이름을 다음 단계에서 사용합니다.

3.  **PSC용 비공개 DNS 영역(Private DNS Zone) 설정**
    *   이 단계가 누락되면 Cloud Run에서 **`Name or service not known`** 오류가 발생합니다.
    *   `goog` DNS 이름으로 비공개 영역을 만들고 VPC에 연결한 후, PSC의 전체 DNS 이름과 IP 주소로 **A 레코드**를 추가해야 합니다.

4.  **데이터베이스 접속 및 생성:**
    *   AlloyDB는 VPC 내부에 있으므로, **동일한 VPC에 연결된 GCE VM**을 통해 접속해야 합니다.
    *   GCE VM에 SSH로 접속한 후, 아래 명령어로 `psql` 클라이언트를 사용하여 접속합니다. `[PSC_DNS_NAME]`을 위에서 확인한 DNS 이름으로 변경하세요.
        ```bash
        # psql 클라이언트가 없다면 설치: sudo apt-get update && sudo apt-get install -y postgresql-client
        psql -h [PSC_DNS_NAME] -U postgres -d postgres
        ```
    *   접속 후, PoC에 사용할 데이터베이스를 생성합니다.
        ```sql
        CREATE DATABASE document_embeddings;
        ```

5.  **`pgvector` 확장 활성화 및 테이블 생성:**
    *   새로 만든 데이터베이스에 다시 접속(`\c document_embeddings`)한 후, 다음 SQL 쿼리를 실행합니다.
        ```sql
        -- pgvector 확장 기능 활성화
        CREATE EXTENSION IF NOT EXISTS vector;

        -- 기존 테이블이 있다면 삭제합니다.
        DROP TABLE IF EXISTS document_embeddings;

        -- 벡터 데이터와 메타데이터를 저장할 테이블 생성
        CREATE TABLE document_embeddings (
            id SERIAL PRIMARY KEY,
            source_file VARCHAR(1024) NOT NULL,
            chunk_description TEXT,
            metadata JSONB,
            text_embedding VECTOR(768),       -- text-multilingual-embedding-002 모델 기준
            multimodal_embedding VECTOR(1408), -- multimodalembedding@001 모델 기준
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX ON document_embeddings USING hnsw (text_embedding vector_l2_ops);
        CREATE INDEX ON document_embeddings USING hnsw (multimodal_embedding vector_l2_ops);
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
    --path-prefix=pdfs/ \
    --event-types=OBJECT_FINALIZE
```

---

### **Phase 2: 데이터 수집 및 처리 서브시스템 (Data Ingestion & Processing)**

#### **1. 서비스 코드 및 의존성 준비**

**`requirements.txt` 파일 준비**

가장 먼저 `main.py`가 위치한 디렉토리에 `requirements.txt` 파일을 생성하고, 서비스에 필요한 라이브러리와 **최신 Vertex AI SDK 버전**을 명시적으로 지정해야 합니다. SDK 버전이 낮으면 `client_options` 같은 중요 파라미터를 인식하지 못해 `TypeError`가 발생합니다.

`~/rag_chatbot/data-ingestion/requirements.txt`
```txt
# 웹 프레임워크
Flask==3.0.0
gunicorn==21.2.0

# GCP 및 데이터베이스
google-cloud-storage>=2.14.0
# [중요] 최신 Vertex AI 기능을 사용하기 위해 버전을 명시적으로 지정합니다.
google-cloud-aiplatform>=1.55.0
pg8000>=1.30.0

# 데이터 처리
pandas>=2.0.0
openpyxl>=3.1.0
PyMuPDF # fitz 모듈 제공
Pillow>=10.0.0
```

**`main.py` 코드 준비 (최종 아키텍처 반영)**

`main.py` 코드는 여러 모델이 각기 다른 최적의 API 엔드포인트에 존재한다는 사실을 반영해야 합니다. `ClientOptions`를 사용하여 각 모델을 초기화할 때 명시적으로 엔드포인트를 지정하는 것이 가장 안정적인 방법입니다.

*   **핵심 원리:**
    *   `gemini-2.5-pro`와 같은 최신 LLM은 `us-central1` 리전에서 가장 먼저 제공되며, 사실상의 글로벌 엔드포인트 역할을 합니다.
    *   `text-multilingual-embedding-002`와 같은 임베딩 모델은 `asia-northeast3`를 포함한 여러 리전에서 안정적으로 제공됩니다.
    *   따라서 각 모델을 **자신이 존재하는 올바른 엔드포인트 주소**로 직접 호출해야 합니다.

*   **`main.py`의 `init_clients` 함수 수정 예시:**
    ```python
    # ClientOptions를 사용하기 위해 추가
    from google.api_core import client_options
    # ... 다른 import 구문

    def init_clients():
        # ...
        try:
            # 1. SDK를 특정 리전 없이 프로젝트만으로 기본 초기화
            vertexai.init(project=PROJECT_ID)

            # 2. Gemini 모델은 'us-central1' 엔드포인트에서 초기화
            gemini_client_options = client_options.ClientOptions(api_endpoint="us-central1-aiplatform.googleapis.com")
            google_search_tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
            gemini_model = GenerativeModel(
                "gemini-2.5-pro",
                tools=[google_search_tool],
                client_options=gemini_client_options
            )

            # 3. 임베딩 모델들은 원래 리전('asia-northeast3') 엔드포인트에서 초기화
            embedding_client_options = client_options.ClientOptions(api_endpoint="asia-northeast3-aiplatform.googleapis.com")
            text_embedding_model = TextEmbeddingModel.from_pretrained(
                "text-multilingual-embedding-002",
                client_options=embedding_client_options
            )
            # ... 다른 모델도 동일하게 client_options 적용 ...
        # ...
    ```

#### **2. 환경 변수 파일 생성 (`env.yaml`)**

`GCP_PROJECT`는 Cloud Run의 기본 환경변수가 아니므로, **반드시 명시적으로 추가해야 합니다.**

`~/rag_chatbot/data-ingestion/env.yaml` 파일을 아래 내용으로 생성하세요.
```yaml
GCP_PROJECT: "[YOUR_PROJECT_ID]"  # [필수] SDK가 프로젝트를 인식하도록 명시적으로 추가
REGION: "asia-northeast3"         # 임베딩 모델 등을 위한 리전 정보
DB_HOST: "[PSC_DNS_NAME]"
DB_USER: "postgres"
DB_PASS: "[YOUR_DB_PASSWORD]"
DB_NAME: "document_embeddings"
```

#### **3. Cloud Run 서비스 배포**

`main.py`가 있는 디렉토리에서 다음 명령어를 실행하여 서비스를 배포합니다.

```bash
cd ~/rag_chatbot/data-ingestion

gcloud run deploy file-processor-service \
    --source . \
    --platform managed \
    --region [REGION] \
    --allow-unauthenticated \
    --vpc-connector [VPC_CONNECTOR] \
    --vpc-egress=all-traffic \
    --env-vars-file=env.yaml \
    --service-account [SERVICE_ACCOUNT_EMAIL] \
    --memory=2Gi
```

*   **`[VPC_CONNECTOR]`**: Cloud Run과 **동일한 리전 및 VPC**에 생성된 서버리스 VPC 액세스 커넥터의 이름입니다.
*   **`[SERVICE_ACCOUNT_EMAIL]`**: 배포에 사용할 서비스 계정입니다. 이 계정은 다음 권한을 필수로 가집니다.
    *   Vertex AI 사용자 (`roles/aiplatform.user`)
    *   Storage 객체 뷰어 (`roles/storage.objectViewer`)
    *   **서버리스 VPC 액세스 사용자 (`roles/vpcaccess.user`)**

*   **[중요]** 배포 시 VPC 커넥터 권한 오류가 발생하면 [트러블슈팅](#q-gcloud-run-deploy-실행-시-vpc-커넥터-오류가-발생합니다) 섹션을 반드시 확인하세요.

#### **4. Pub/Sub 구독 생성**

GCS 알림 토픽과 `file-processor-service`를 연결하는 Push 구독을 생성합니다.

```bash
gcloud pubsub subscriptions create gcs-file-event-subscription \
    --topic gcs-file-events \
    --push-endpoint=$(gcloud run services describe file-processor-service --platform managed --region asia-northeast3 --format "value(status.url)") \
    --push-auth-service-account=[CLOUD_RUN_INVOKER_SERVICE_ACCOUNT_EMAIL]
```

---

### **Phase 3: 백필(Backfill) 및 재처리 테스트**

`file-processor-service`의 코드를 수정했거나, 특정 파일 처리가 누락/실패했을 때, GCS에 파일을 다시 업로드하지 않고도 기존 파일들을 재처리할 수 있습니다. `backfill.py` 스크립트는 GCS 버킷의 특정 폴더에 있는 모든 파일 목록을 읽어, 각 파일에 대해 GCS 업로드 이벤트(Pub/Sub 메시지)를 수동으로 생성하여 파이프라인을 트리거합니다.

이 과정은 로컬 PC 또는 Cloud Shell에서 Python 가상 환경을 설정하여 안전하게 실행합니다.

#### **1. 백필 스크립트 준비 (`backfill.py`)**

먼저 `~/rag_chatbot/backfill/` 디렉토리에 필요한 파일들이 있는지 확인합니다.

**`requirements.txt`**
이 디렉토리에는 Python 라이브러리 의존성을 정의한 `requirements.txt` 파일이 있어야 합니다.

```txt
google-cloud-storage
google-cloud-pubsub
```

**`backfill.py`**
스크립트는 GCS와 Pub/Sub 클라이언트를 사용하여 메시지를 발행하는 로직을 포함해야 합니다. (아래는 기능 이해를 돕기 위한 예시 코드)

```python
import argparse
import json
from google.cloud import storage, pubsub_v1

def trigger_gcs_events(project_id, topic_id, gcs_bucket, gcs_prefix):
    """Lists files in GCS and publishes a Pub/Sub message for each."""
    
    storage_client = storage.Client(project=project_id)
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    blobs = storage_client.list_blobs(gcs_bucket, prefix=gcs_prefix)
    
    print(f"Found files in gs://{gcs_bucket}/{gcs_prefix}. Publishing events to topic '{topic_id}'...")
    
    for blob in blobs:
        # GCS OBJECT_FINALIZE 이벤트와 유사한 메시지 구조 생성
        message_data = {
            "bucket": gcs_bucket,
            "name": blob.name
        }
        
        # 데이터를 JSON 문자열로 변환 후, 바이트로 인코딩
        message_bytes = json.dumps(message_data).encode("utf-8")
        
        future = publisher.publish(topic_path, data=message_bytes)
        print(f"Published message for gs://{gcs_bucket}/{blob.name}, message_id: {future.result()}")

    print("Backfill process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCS backfill script for Pub/Sub.")
    parser.add_argument("--project_id", required=True, help="Your Google Cloud project ID.")
    parser.add_argument("--topic_id", required=True, help="The Pub/Sub topic ID to publish to.")
    parser.add_argument("--gcs_bucket", required=True, help="The GCS bucket name.")
    parser.add_argument("--gcs_prefix", default="", help="Optional GCS path prefix to filter files.")
    
    args = parser.parse_args()
    
    trigger_gcs_events(args.project_id, args.topic_id, args.gcs_bucket, args.gcs_prefix)
```

#### **2. 로컬/Cloud Shell 환경 설정 및 실행**

1.  **디렉토리 이동:**
    Cloud Shell 또는 로컬 터미널에서 `backfill.py`가 있는 곳으로 이동합니다.
    ```bash
    cd ~/rag_chatbot/backfill
    ```

2.  **Python 가상 환경 생성 및 활성화:**
    프로젝트별로 독립된 라이브러리 환경을 사용하기 위해 가상 환경을 생성합니다.
    ```bash
    # 가상 환경 'venv' 생성
    python3 -m venv venv

    # 가상 환경 활성화 (터미널 프롬프트 앞에 '(venv)'가 나타남)
    source venv/bin/activate
    ```

3.  **필요 라이브러리 설치:**
    가상 환경이 **반드시 활성화된 상태**에서 `requirements.txt` 파일로 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

4.  **GCP 인증:**
    스크립트가 GCP 서비스(GCS, Pub/Sub)에 접근할 수 있도록 로컬 환경에서 사용자 계정으로 인증합니다.
    ```bash
    gcloud auth application-default login
    ```
    이 명령어는 웹 브라우저를 열어 Google 계정으로 로그인하도록 요청하며, 인증 정보를 로컬에 저장합니다.

5.  **백필 스크립트 실행:**
    이제 모든 준비가 끝났습니다. 아래 명령어를 실행하여 백필을 시작합니다. `[PLACEHOLDER]` 부분들을 실제 값으로 변경하세요.
    ```bash
    python backfill.py \
        --project_id [YOUR_PROJECT_ID] \
        --gcs_bucket [YOUR_GCS_BUCKET_NAME] \
        --gcs_prefix pdfs/ \
        --topic_id gcs-file-events
    ```

#### **3. 결과 모니터링**

스크립트가 실행되면 터미널에 각 파일에 대한 메시지 발행 로그가 출력됩니다. 이와 동시에, **Cloud Run 콘솔의 로그 탐색기**로 이동하여 `file-processor-service`의 로그를 확인하세요. GCS의 각 파일에 대해 서비스가 순차적으로 트리거되며 처리 로그가 나타나는 것을 볼 수 있습니다. 이를 통해 전체 데이터 처리 파이프라인이 정상적으로 작동하는지 검증할 수 있습니다.

#### **4. 가상 환경 비활성화**

테스트가 끝나면 아래 명령어로 가상 환경을 빠져나옵니다.
```bash
deactivate
```

---

### **Phase 4: 서빙 서브시스템 (Serving Subsystem)**

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

### **Phase 5: 품질 평가 서브시스템 (Quality Evaluation)**

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
CREATE OR REPLACE TABLE `[YOUR_PROJECT_ID].[YOUR_DATASET_ID].evaluation_results` (
    prompt STRING OPTIONS(description="에이전트에게 보낸 질문 또는 프롬프트"),
    response STRING OPTIONS(description="에이전트가 생성한 답변"),
    evaluation_score INT64 OPTIONS(description="자동 평가 모델이 채점한 점수"),
    evaluation_reason STRING OPTIONS(description="점수에 대한 평가 근거"),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
);
```

#### **3. 평가 실행**

*   **Cloud Run Job으로 배포:** 위 코드를 Job 형태로 배포하여 필요할 때마다 실행할 수 있습니다.
*   **Trigger 설정:** Cloud Scheduler와 Pub/Sub을 사용하여 정기적으로(예: 매일 밤) 평가 작업을 트리거하도록 구성할 수 있습니다.

---

### **트러블슈팅 (Troubleshooting)**

#### Q: GCE VM에서 AlloyDB로 `psql` 접속이 안 됩니다 (I/O Timeout 등).

**A:** 이것은 대부분 PSC 연결 문제입니다. 아래 순서대로 확인하세요.
1.  **PSC 엔드포인트 상태 확인:** AlloyDB 인스턴스의 '연결' 탭에서 PSC 엔드포인트 상태가 `✅ 수락됨`이고, 내부 IP가 할당되었는지 확인하세요. `⚠️ 확인 필요` 상태라면 다음 단계를 진행하세요.
2.  **Service Connection Policy 확인:** `확인 필요` 상태의 원인은 조직의 보안 정책(`ServiceConnectionPolicy`)이 연결을 막고 있기 때문입니다. 조직 관리자 권한으로 아래 명령어를 실행하여 `dev-vpc` 네트워크가 `alloydb.googleapis.com` 서비스에 접속할 수 있도록 허용하는 정책을 생성해야 합니다.
    ```bash
    gcloud network-connectivity service-connection-policies create allow-alloydb-for-dev-vpc \
        --service-class=alloydb.googleapis.com \
        --network=projects/[YOUR_PROJECT_ID]/global/networks/dev-vpc \
        --consumer-resource-roots=projects/[YOUR_PROJECT_ID] \
        --location=global \
        --project=[YOUR_PROJECT_ID]
    ```
3.  **정책 생성 후 연결 수락:** 위 명령어가 성공하면, 다시 AlloyDB 콘솔로 돌아가 PSC 연결 요청을 **수락**합니다.
4.  **DNS 이름으로 접속:** 반드시 할당된 IP가 아닌, 긴 **DNS 이름**을 사용하여 `psql -h [PSC_DNS_NAME] ...` 명령으로 접속해야 합니다.

#### Q: GCE VM에 SSH 접속이 안 됩니다 (웹 브라우저, gcloud 모두 먹통).

**A:** VPC의 방화벽 문제일 가능성이 99%입니다.
1.  **SSH 허용 방화벽 규칙 생성:** Cloud Shell에서 아래 명령어를 실행하여 인터넷에서 VM으로의 SSH(TCP 포트 22) 접속을 허용하는 방화벽 규칙을 생성합니다. `[VPC_NETWORK_NAME]`을 `dev-vpc` 등으로 변경하세요.
    ```bash
    gcloud compute firewall-rules create default-allow-ssh \
        --network=[VPC_NETWORK_NAME] \
        --allow=tcp:22 \
        --source-ranges=0.0.0.0/0
    ```
2.  **외부 IP 확인:** VM에 외부 IP가 할당되어 있는지 확인하세요. 없다면 VM을 수정하여 임시 외부 IP를 할당해야 합니다.

#### Q: `gcloud run deploy` 실행 시 VPC 커넥터 오류가 발생합니다.

**A:** 오류 메시지가 `VPC connector ... does not exist, or Cloud Run does not have permission to use it` 라면, 권한 문제입니다.
1.  **커넥터 존재 확인:** `gcloud compute networks vpc-access connectors list --region asia-northeast3` 명령으로 커넥터가 `READY` 상태로 존재하는지 먼저 확인합니다.
2.  **두 서비스 계정에 권한 부여:** **두 개의** 서비스 계정에 `roles/vpcaccess.user` 역할을 부여해야 합니다.
    *   **사용자 지정 서비스 계정:**
        ```bash
        gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] \
            --member="serviceAccount:[SERVICE_ACCOUNT_EMAIL]" \
            --role="roles/vpcaccess.user"
        ```
    *   **Cloud Run 서비스 에이전트 (Google 관리):**
        ```bash
        gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] \
            --member="serviceAccount:service-[YOUR_PROJECT_NUMBER]@serverless-robot-prod.iam.gserviceaccount.com" \
            --role="roles/vpcaccess.user"
        ```

#### Q: `gcloud run deploy` 실행 중 `gcloud crashed (TypeError)` 오류가 발생합니다.

**A:** `gcloud` 도구 자체의 문제 또는 빌드 환경의 네트워크 문제입니다.
1.  **`gcloud` 업데이트:** `gcloud components update`를 실행하여 도구를 최신화합니다.
2.  **`env.yaml` 사용:** 명령어에 직접 환경 변수를 주입하는 대신, 본문에 안내된 `env.yaml` 파일을 사용하는 `--env-vars-from-file` 방식으로 변경하세요. 이 방법이 훨씬 안정적입니다.
3.  **Cloud Build 권한 확인:** IAM 페이지에서 `[PROJECT_NUMBER]@cloudbuild.gserviceaccount.com` 서비스 계정에 `Cloud Run 관리자` 및 `서비스 계정 사용자` 역할이 있는지 확인하세요.

#### Q: Cloud Run 로그에 `Name or service not known` 또는 `Can't create a connection to host` 오류가 발생합니다.

**A:** Cloud Run이 AlloyDB의 PSC DNS 주소를 IP로 변환하지 못하는, 전형적인 **비공개 DNS 조회 문제**입니다.
1.  **가장 먼저 [Phase 1의 3단계](#3-가장-중요-psc용-비공개-dns-영역private-dns-zone-설정)를 수행했는지 확인하세요.** `goog` DNS 이름으로 비공개 DNS 영역을 만들고, VPC 네트워크에 연결한 후, PSC의 전체 DNS 이름과 IP 주소로 A 레코드를 추가해야 합니다.
2.  Cloud DNS API(`dns.googleapis.com`)가 활성화되어 있는지 확인하세요.

#### Q: Cloud Run 로그에 `404 Not Found ... models/gemini-2.5-pro` 오류가 발생합니다.

**A:** API를 호출한 리전(`asia-northeast3`)에 해당 모델이 없기 때문입니다.
1.  `main.py`의 `init_clients` 함수에서 `vertexai.init(location="global")`로 설정하여 **글로벌 엔드포인트**를 사용하도록 수정하세요.

#### Q: `gcloud run deploy` 실행 시 VPC 커넥터 오류가 발생합니다.

**A:** 오류 메시지가 `VPC connector ... does not exist, or Cloud Run does not have permission to use it` 라면, 권한 문제입니다.
1.  **두 개의** 서비스 계정에 `roles/vpcaccess.user` 역할을 부여해야 합니다.
    *   **배포에 사용한 서비스 계정:** `[SERVICE_ACCOUNT_EMAIL]`
    *   **Cloud Run 서비스 에이전트:** `service-[YOUR_PROJECT_NUMBER]@serverless-robot-prod.iam.gserviceaccount.com`

#### Q: Cloud Run 로그에 `404 Not Found ... Publisher Model ... is not found` 오류가 발생합니다.

**A:** API를 호출한 엔드포인트에 해당 모델이 존재하지 않기 때문입니다. 이는 **모든 모델이 동일한 리전(`asia-northeast3`)이나 `global` 엔드포인트에 있지 않기 때문**에 발생합니다.
1.  **해결책:** `main.py`의 `init_clients` 함수에서, 각 모델을 생성할 때 `client_options`를 사용하여 올바른 API 엔드포인트를 명시적으로 지정해야 합니다.
2.  **예시:**
    *   `gemini-2.5-pro` -> `api_endpoint="us-central1-aiplatform.googleapis.com"`
    *   `text-multilingual-embedding-002` -> `api_endpoint="asia-northeast3-aiplatform.googleapis.com"`
3.  자세한 코드는 **[Phase 2의 1단계](#1-cloud-run-서비스-코드-준비-최종-아키텍처-반영)**를 참조하세요.

#### Q: Cloud Run 로그에 `Initializing Vertex AI for project 'None'...`가 보이며, 이후 404 오류가 발생합니다.

**A:** Cloud Run 환경에 `GCP_PROJECT` 환경 변수가 설정되지 않아 SDK가 어떤 프로젝트로 API를 호출해야 할지 모르기 때문입니다.
1.  **해결책:** `env.yaml` 파일에 `GCP_PROJECT: "[YOUR_PROJECT_ID]"` 한 줄을 **반드시 추가**한 후, 다시 배포하세요.

#### Q: Cloud Run 로그에 `Name or service not known` 또는 `Can't create a connection to host` 오류가 발생합니다.

**A:** Cloud Run이 AlloyDB의 PSC DNS 주소를 IP로 변환하지 못하는, 전형적인 **비공개 DNS 조회 문제**입니다.
1.  **해결책:** **[Phase 1의 2단계](#2-alloydb-for-postgresql-설정)**에 있는 **PSC용 비공개 DNS 영역 설정** 가이드를 다시 한번 꼼꼼히 확인하고 그대로 실행하세요. `goog` DNS 이름으로 비공개 영역을 만들고 VPC에 연결한 후, A 레코드를 추가해야 합니다.
