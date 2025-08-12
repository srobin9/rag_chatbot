# backfill.py
from google.cloud import storage, pubsub_v1
import json
import traceback

# --- 설정 (자신의 환경에 맞게 수정) ---
PROJECT_ID = "p-khm8-dev-svc"
BUCKET_NAME = "p-khm8-dev-svc-samsungena"
TOPIC_NAME = "gcs-file-events"
PREFIX = "pdfs/"  # 처리할 폴더 지정

def backfill_gcs_events():
    """GCS 버킷의 파일 목록을 읽어 Pub/Sub에 이벤트를 직접 게시합니다."""
    publisher = None  # try 블록 밖에서 변수 선언
    try:
        storage_client = storage.Client()
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

        blobs_iterator = storage_client.list_blobs(BUCKET_NAME, prefix=PREFIX)
        
        print("--- Backfill Script Start ---")
        
        # 이터레이터에서 파일 목록을 한 번만 추출하여 리스트로 저장합니다.
        files_to_process = [blob for blob in blobs_iterator if not blob.name.endswith('/')]
        total_files = len(files_to_process)

        if total_files == 0:
            print(f"No files found in gs://{BUCKET_NAME}/{PREFIX}")
            return

        print(f"Found {total_files} files to process.")

        # 생성된 리스트(files_to_process)를 사용하여 루프를 실행합니다.
        for i, blob in enumerate(files_to_process):
            print(f"[{i + 1}/{total_files}] Publishing message for: {blob.name}")
            
            # Cloud Run 코드가 기대하는 메시지 형식 생성
            message_data = {
                "bucket": BUCKET_NAME,
                "name": blob.name
            }
            
            # 데이터를 JSON 문자열로 변환 후 바이트로 인코딩
            data_to_publish = json.dumps(message_data).encode("utf-8")
            
            # Pub/Sub에 메시지 게시
            future = publisher.publish(topic_path, data=data_to_publish)
            
            # 메시지가 성공적으로 게시될 때까지 동기적으로 대기
            future.result() 
            print(f" -> Successfully published.")

        print(f"\n--- Backfill Complete: {total_files} messages published. ---")
        print("Please check your Cloud Run logs to see the results.")

    except Exception as e:
        # 오류 발생 시 더 상세한 정보 출력
        print("\n--- SCRIPT FAILED ---")
        print(f"An unexpected error occurred: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")

if __name__ == "__main__":
    backfill_gcs_events()