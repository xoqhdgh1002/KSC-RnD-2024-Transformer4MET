#!/usr/bin/env bash
#SBATCH -J diffmet
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/%x-%j-out.log
#SBATCH -e ./logs/%x-%j-err.log
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

echo "START: $(date)"

echo "PROJECT_PREFIX=${PROJECT_PREFIX}"
if [ -z "${PROJECT_PREFIX}" ]; then
    echo "PROJECT_PREFIX is not defined. pelase source setup.sh" 1>&2
    exit 1
fi

export OMP_NUM_THREADS=1

echo "MAMBA_EXE=${MAMBA_EXE}"
eval "$(${MAMBA_EXE} shell hook --shell=bash)"
micromamba activate diffmet-py311

python train.py \
    --data ./config/data/test/test.yaml \
    --model ./config/model/test/test.yaml \
    --trainer ./config/trainer/main.yaml \
    --optimizer ./config/optimizer/AdamW.yaml

echo "END: $(date)"
exit 0
