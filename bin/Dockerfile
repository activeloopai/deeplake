FROM python:3.8-slim

RUN apt-get -y update && \
    apt-get -y install git wget build-essential python-setuptools python3-dev libjpeg-dev libpng-dev zlib1g-dev libcurl4-openssl-dev && \
    apt install build-essential

RUN mkdir /app

RUN git clone https://github.com/activeloopai/Hub.git /app/
WORKDIR /app

RUN pip install -e .[all]

RUN pip install -r deeplake/requirements/plugins.txt && \
    pip install -r deeplake/requirements/tests.txt