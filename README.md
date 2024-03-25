# Differential MET Reconstruction

## Recipes
### Install
#### Micromamba
copied from [Micromamba Installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install)
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

#### oython environment
```bash
micromamba create -y -f environment.yaml
```

### Setup
if you are on bash
```bash
source setup.sh
```

### SubmitBatch job with slurm
```bash
mkdir logs # directory for slurm's output
sbatch ./run.sh
squeue -u $USER
```
