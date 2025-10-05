# Use Python 3.10 slim image as base
# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by OpenCV, pdf2image and pytesseract
# - libgl1, libglvnd0: provides libGL.so.1 for OpenCV
# - libglib2.0-0, libsm6, libxrender1, libxext6: common X11 deps for cv2 and plotting
# - poppler-utils: provides pdftoppm for pdf2image
# - tesseract-ocr and tesseract-ocr-eng: OCR engine and English language pack
# - build-essential and pkg-config: for compiling any native wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    curl \
    libgl1 \
    libglvnd0 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Docker-optimized requirements first for better cache utilization
# Copy requirements and install Python dependencies (use cache when possible)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the application code
COPY . .

# Create a non-root user for security (after installing packages)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]