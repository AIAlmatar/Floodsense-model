FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict_and_insert.py .
COPY models ./models

RUN mkdir -p outputs

CMD ["python", "predict_and_insert.py"]