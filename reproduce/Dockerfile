# Based on https://github.com/jupyter/docker-stacks/blob/master/datascience-notebook/Dockerfile
ARG BASE_CONTAINER=jupyter/scipy-notebook:6d42503c684f
FROM $BASE_CONTAINER

LABEL maintainer="Mridul Seth <seth.mridul@gmail.com>"

USER root

# install latex-full installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends texlive-full dvipng && \
    rm -rf /var/lib/apt/lists/*

USER $NB_UID
WORKDIR $HOME

# install the required packages for pandemic repo.
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
