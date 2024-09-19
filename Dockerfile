FROM python:3-slim-bullseye

WORKDIR /opt

ADD requirements.txt ./

RUN pip install -r requirements.txt && \
    apt-get update && \
    apt-get install -y wget && \
    wget http://cloudpricingcalculator.appspot.com/static/data/pricelist.json

ADD monitor.py ./

ADD gcp_monitor.py ./

ENTRYPOINT ["./monitor.py"]
