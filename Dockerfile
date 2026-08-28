FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV \
    DATABASE_URL=postgresql://nba:nba@db:5432/nba \
    NBA_DB_SCHEMA=bronze \
    NBA_DB_SCHEMAS=bronze,silver,gold \
    NBA_AUTO_CREATE_SCHEMAS=false \
    MAX_QUERY_ROWS=5000

COPY app.py db.py query_engine.py lightweight_rag.py ./
COPY .streamlit ./.streamlit
COPY knowledge ./knowledge
COPY scripts ./scripts

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
