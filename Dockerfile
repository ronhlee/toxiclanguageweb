FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY models models/

COPY templates templates/

COPY static static/

ADD app.py /

EXPOSE 8080

CMD ["python", "./app.py"]
