# 베이스 이미지 선택
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    python3-tk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./3D_costmap .

CMD ["bash"]