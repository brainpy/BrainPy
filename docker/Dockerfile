FROM ubuntu:22.04

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install -y --no-install-recommends software-properties-common

RUN apt update && apt install -y python3-pip

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip


RUN pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install --upgrade setuptools && \
    pip --no-cache-dir install --upgrade wheel

ADD . /usr/src/app
WORKDIR /usr/src/app

RUN pip --no-cache-dir install --upgrade "jax[cpu]"
RUN pip --no-cache-dir install --upgrade -r requirements.txt
