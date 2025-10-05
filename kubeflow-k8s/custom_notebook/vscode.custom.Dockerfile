FROM ghcr.io/kubeflow/kubeflow/notebook-servers/codeserver-python:v1.10.0

ARG CONDA_ENV="dev"
ARG PYTHON_VERSION_DEV=3.12

USER root

RUN sed -i -E "s|^\s*conda activate\s+.*|conda activate ${CONDA_ENV}|" ${HOME_TMP}/.bashrc \
    && sed -i -E "s|^\s*conda activate\s+.*|conda activate ${CONDA_ENV}|" /etc/profile
    
COPY --chown=${NB_USER}:${NB_GID} dev.requirements.txt /tmp

USER $NB_UID

RUN conda create -y -q \
    python=${PYTHON_VERSION_DEV} \
    --name ${CONDA_ENV} \
    && conda update -y -q --all \
    && conda clean -a -f -y

RUN source ${HOME_TMP}/.bashrc \
    && pip3 install --no-cache-dir -r /tmp/dev.requirements.txt
    
RUN mkdir ${HOME_TMP}/app
ADD git@github.com:urospes/kubeflow-playground.git ${HOME_TMP}/app

