# 1. 베이스 이미지 선택: Google Cloud에서 권장하는 Python 3.11 slim 버전을 사용합니다.
FROM python:3.11-slim

# 2. 환경 변수 설정
# - PYTHONUNBUFFERED: Python의 출력 버퍼를 비활성화하여 로그가 즉시 Cloud Logging에 표시되도록 합니다.
# - APP_HOME: 애플리케이션 코드가 위치할 컨테이너 내부 디렉토리를 정의합니다.
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME

# 3. 시스템 의존성 설치 (Build-time)
# 'slim' 이미지는 매우 최소화되어 있으므로, C/C++ 확장을 컴파일해야 할 수 있는
# 라이브러리(예: PyMuPDF, 일부 gRPC 의존성)를 위해 빌드 도구를 설치합니다.
# --no-install-recommends는 불필요한 패키지 설치를 막고, 설치 후에는 apt 캐시를 삭제하여 이미지 크기를 최적화합니다.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# 4. Python 의존성 설치
# 먼저 requirements.txt 파일만 복사하여 Docker의 레이어 캐시를 효율적으로 활용합니다.
COPY requirements.txt .

# 안정성 강화: pip를 업그레이드하고 requirements.txt를 한 번에 설치하여 레이어를 최적화하고 캐시 문제를 방지합니다.
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "--- INSTALLED PACKAGES (pip list) ---" && \
    pip list && \
    echo "--- DEPENDENCY CHECK (pip check) ---" && \
    pip check

# 빌드 검증 단계: 실제 코드에서 사용하는 핵심 모듈을 직접 임포트하여 검증합니다.
# 이 명령어가 성공하면, main.py에서도 임포트가 성공할 확률이 매우 높습니다.
RUN python -c "from google.cloud.aiplatform.language_models import TextEmbeddingModel; print('Embedder Service SDK verification successful.')"

# 5. 소스 코드 복사
# 나머지 애플리케이션 코드를 컨테이너로 복사합니다.
COPY . .

# 6. 애플리케이션 실행
# Cloud Run이 주입하는 PORT 환경 변수를 사용하여 Gunicorn 프로덕션 서버를 실행합니다.
# --workers: 동시 요청 처리 능력을 높이기 위해 2로 설정합니다. (Cloud Run CPU 설정에 따라 조절)
# --threads: 각 워커가 처리할 스레드 수입니다. I/O 바운드 작업에 효과적입니다.
# --timeout: 워커가 응답 없이 대기할 수 있는 최대 시간(초)입니다. 0은 무한 대기를 의미하며,
# Cloud Run의 요청 시간 제한(최대 3600초)보다 짧게 설정하는 것이 좋습니다.
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 600 main:app