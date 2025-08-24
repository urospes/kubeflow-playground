FROM ghcr.io/kubeflow/kubeflow/notebook-servers/codeserver-python:v1.10.0

ARG CONDA_ENV="dev"
ARG PYTHON_VERSION_DEV=3.12

USER $NB_UID

RUN conda create -y -q \
    python=${PYTHON_VERSION_DEV} \
    --name ${CONDA_ENV} \
    && conda update -y -q --all \
    && conda clean -a -f -y

USER root

RUN echo "conda activate $CONDA_ENV" >> ${HOME}/.bashrc

USER $NB_UID

COPY --chown=${NB_USER}:${NB_GID} dev_env_requirements.txt /tmp

RUN source ~/.bashrc \
    && pip3 install --no-cache-dir -r /tmp/dev_env_requirements.txt