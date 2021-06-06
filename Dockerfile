FROM python:3.8-slim

RUN apt-get -y update && apt-get -y install git wget build-essential python-setuptools python3-dev

RUN mkdir /app

RUN git clone https://github.com/activeloopai/Hub.git /app/ && \
    cd /app/ && \
    git checkout clean_2

WORKDIR /app

RUN pip install -r requirements/requirments.txt && \
    pip install -r requirements/common.txt && \
    pip install -r requirements/tests.txt && \
    pip install -r requirements/plugins.txt

RUN pytest .

ENV PYTHONPATH="/app/Hub:$PYTHONPATH"
