FROM python:3-slim-bullseye

WORKDIR /opt

ADD requirements.txt ./
RUN apk add --no-cache \
  build-base \
  libstdc++ \
  linux-headers \
  && \
  pip install -r requirements.txt && \
  apk del \
  build-base \
  linux-headers
ADD monitor.py ./
ADD gcp_monitor.py ./

ENTRYPOINT ["./monitor.py"]
