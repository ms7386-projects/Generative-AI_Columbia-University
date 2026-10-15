FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# CPU wheels keep inference portable; no Mac MPS/CUDA dependency in the container.
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_md
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
