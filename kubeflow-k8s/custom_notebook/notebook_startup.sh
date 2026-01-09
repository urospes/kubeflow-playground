#!/bin/bash
set -eu
cd ${HOME_TMP}
git config --global user.email "user@example.com"
git config --global user.name "kubeflow notebook"
git clone git@github.com:urospes/kubeflow-playground.git
exec /init