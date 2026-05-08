FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/      agent/
COPY aws/        aws/
COPY data/       data/
COPY model/      model/

CMD ["python", "aws/worker.py"]
