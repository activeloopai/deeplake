FROM python:3.6
ADD ./ /workspace
WORKDIR /workspace
RUN pip install -r optionals.txt
RUN pip install -e /workspace