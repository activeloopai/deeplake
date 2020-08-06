FROM python:3.6
ADD ./ /workspace
WORKDIR /workspace
RUN pip install -r requirements_dev.txt
RUN pip install -e /workspace