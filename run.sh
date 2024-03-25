#!/bin/bash
#SBATCH -J diffmet
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/%x-%j-out.log
#SBATCH -e ./logs/%x-%j-err.log
#SBATCH --time 06:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

echo "START: $(date)"

echo "PROJECT_PREFIX=${PROJECT_PREFIX}"
if [ -z "${PROJECT_PREFIX}" ]; then
    echo "PROJECT_PREFIX is not defined. pelase source setup.sh" 1>&2
    exit 1
fi

export OMP_NUM_THREADS=1

export PATH=${HOME}/.local/bin/:${PATH}
eval "$(micromamba shell hook --shell=bash)"
micromamba activate diffmet-py311

CONFIG_FILE=${PROJECT_PREFIX}/config/test-l1pf-transformer-neuron.yaml
echo "CONFIG_FILE=${CONFIG_FILE}"
if [ ! -f ${CONFIG_FILE} ]; then
    echo "CONFIG_FILE not found" 1>&2
fi

python run.py -c ${CONFIG_FILE}

echo "END: $(date)"
exit 0
