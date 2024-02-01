FROM python:3.11.7-slim-bullseye

WORKDIR /app

COPY ./ ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python3", "main.py"]
