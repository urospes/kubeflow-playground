#!/bin/bash
set -eu
cd ${HOME_TMP}
ls -al .
cat /home/jovyan/.ssh/known_hosts
cat /home/jovyan/.ssh/id_ed25519
git config --global user.email "user@example.com"
git config --global user.name "kubeflow notebook"
git clone git@github.com:urospes/kubeflow-playground.git
exec /init