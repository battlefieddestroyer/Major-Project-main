FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirement.txt

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.enableCORS=false"]