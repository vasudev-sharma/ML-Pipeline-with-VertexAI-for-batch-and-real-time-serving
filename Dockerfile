# Base Docker image
# FROM ubuntu:22.04 as base
FROM python:3.9-slim 

# Install python 3.9
# ENV PYTHONUNBUFFERED=1
# RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
# RUN python3 -m ensurepip -- issues with it
# RUN pip3 install --no-cache --upgrade pip setuptools
# RUN apk update
# RUN apt update && apt -y install software-properties-common
# RUN apt install python3.9



# Install dependencies
COPY requirements/requirements.in requirements.in
RUN pip install pip-tools && pip-compile -r requirements.in && pip install -r requirements.txt

# COPY inference script and pipeline config file
COPY inference_script.py inference_script.py
COPY pipeline_script.py pipeline_script.py
COPY configs /configs
COPY configs/pipeline_config.yaml pipeline_config.yaml


COPY src /src
ENTRYPOINT [ "python",  "-m", "src.training_script" ]