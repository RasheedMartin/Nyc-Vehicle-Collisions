FROM python:3.12-slim

WORKDIR /app

# System deps for geopandas
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY app/ ./app/

# Streamlit config
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml

EXPOSE 8501

# CMD is overridden by railway.toml startCommand in production
# For local docker run: docker run -e PORT=8501 ...
CMD ["sh", "-c", "streamlit run app/main.py --server.address=0.0.0.0 --server.headless=true"]