#!/usr/bin/env bash
export PROJECT_PREFIX=$(realpath $(dirname ${BASH_SOURCE}))
echo $PROJECT_PREFIX
export PYTHONPATH=${PROJECT_PREFIX}/src:${PYTHONPATH}
echo $PYTHONPATH
# micromamba activate diffmet-py311
