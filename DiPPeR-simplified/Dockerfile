# 베이스 이미지 선택
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# --- 모든 설치 작업은 root 권한으로 먼저 수행 ---
# root 권한 상태 (기본값)
RUN apt-get update && apt-get install -y \
    python3-tk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 패키지 설치도 root 권한으로 시스템 전체에 설치
RUN pip install --no-cache-dir matplotlib tqdm numpy

# 작업 디렉토리 설정
WORKDIR /usr/src/app

# 소스 코드 복사
COPY ./app .

# --- 모든 설치와 설정이 끝난 후, 보안을 위해 일반 사용자로 전환 ---
# 애플리케이션 실행 전에 사용자를 변경합니다.
USER 1000

# 컨테이너 실행 시 기본 명령어 설정
CMD ["python3", "main.py"]