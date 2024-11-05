FROM python:3-slim-bullseye

WORKDIR /opt

ADD requirements.txt ./
RUN pip install -r requirements.txt
ADD monitor.py ./
ADD gcp_monitor.py ./

ENTRYPOINT ["./monitor.py"]
