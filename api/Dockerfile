FROM python:3.10

WORKDIR /app

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ["python", "app.py"]