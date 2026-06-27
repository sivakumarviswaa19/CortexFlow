FROM python:3.13

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]