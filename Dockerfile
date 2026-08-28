# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg is required by librosa for MP3/audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies. torch + torchaudio 를 같은 CPU 인덱스에서 함께 설치해
# 버전을 짝 맞추고, torchaudio 가 의존성으로 CUDA용 torch 를 끌어와 이미지가 부풀거나
# CPU torch 를 덮어쓰는 것을 방지한다. (이후 requirements 의 torch/torchaudio 는 이미 충족되어 건너뜀)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Demucs 가중치를 빌드 시 미리 받아 이미지에 구워넣는다(torch hub 캐시).
# → 런타임 첫 분리 요청에서 수백MB 다운로드 대기가 사라진다. (이미지 크기는 그만큼 증가)
RUN python -c "from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as B; B.get_model()"

# Copy application code
COPY . .

# Create outputs and uploads directories
RUN mkdir -p data/uploads data/outputs

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
