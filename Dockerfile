FROM python:3.9-slim

WORKDIR /app

COPY [requirement.txt](http://_vscodecontentref_/3) [requirement.txt](http://_vscodecontentref_/4)
RUN pip install -r [requirement.txt](http://_vscodecontentref_/5)

COPY . .

CMD ["python", "app.py"]