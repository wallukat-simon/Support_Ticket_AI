FROM python:3.11-slim

WORKDIR /app

# Install UI requirements
COPY requirements_ui.txt .
RUN pip install --no-cache-dir -r requirements_ui.txt

# Copy UI source
COPY src/ui.py .

# API URL will be injected by docker-compose
ENV API_URL=http://api:8000/predict

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
