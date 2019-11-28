FROM amd64/python:3.6 AS build
ADD ./ /workspace
WORKDIR /workspace
RUN pip install -e /workspace 

FROM build as dev
RUN echo $(pip --version)
RUN pip install torch