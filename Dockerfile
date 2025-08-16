# dockerfile

FROM python:3.11-slim

# System deps (kept minimal)
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Install Python deps first for better layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy 
COPY streamlit_app.py ingestion.py ./
# Copy the three artifacts
# COPY processed_documents_clean.json embeddings.npy id_index.json ./

# Create non-root user and a writable logs dir
RUN useradd --create-home appuser \
 && mkdir -p /app/logs \
 && chown -R appuser:appuser /app
USER appuser

# Streamlit runs on 8503, bind to all interfaces
EXPOSE 8503

# Default command (compose can override if you like)
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8503", "--server.address=0.0.0.0"]
