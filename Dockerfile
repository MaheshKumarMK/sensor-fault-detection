FROM python:3.8.5-slim-buster

WORKDIR /main

COPY . /main

RUN pip install -r requirements.txt

CMD ["python3", "main.py"]