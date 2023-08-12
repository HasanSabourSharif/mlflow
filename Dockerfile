FROM python:3.11

WORKDIR /app

COPY ./requirements.txt .

RUN python3 -m pip install -r ./requirements.txt

RUN pip install mlflow
EXPOSE 80

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "80"]
# uvicorn main:app --host 0.0.0.0 --port 80