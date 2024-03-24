#!/usr/bin/env bash
export PROJECT_PREFIX=$(dirname ${BASH_SOURCE})
export PYTHONPATH=${PROJECT_PREFIX}/src:${PYTHONPATH}
micromamba activate diffmet-py311
