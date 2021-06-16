FROM python:3.8-slim

RUN apt-get -y update && \
    apt-get -y install git wget build-essential python-setuptools python3-dev libjpeg-dev libpng-dev zlib1g-dev && \
    apt install build-essential

RUN mkdir /app

RUN git clone https://github.com/activeloopai/Hub.git /app/ && \
    cd /app/ && \
    git checkout release/2.0

WORKDIR /app

RUN pip install -r requirements/common.txt && \
    pip install -r requirements/tests.txt && \
    pip install -r requirements/plugins.txt

ENV PYTHONPATH="/app/:$PYTHONPATH"
RUN python3 -m pip install -U .
