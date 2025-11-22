FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY cost_aware_planner/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY cost_aware_planner/ .

CMD ["bash"]
