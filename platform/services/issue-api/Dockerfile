FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY platform/shared /app/platform/shared
COPY platform/services/issue-api /app/platform/services/issue-api
RUN pip install /app/platform/shared /app/platform/services/issue-api
COPY data/processed/team_mapping.csv /app/data/processed/team_mapping.csv
COPY data/processed/issue_mapping.csv /app/data/processed/issue_mapping.csv
EXPOSE 8000
CMD ["uvicorn", "issue_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
