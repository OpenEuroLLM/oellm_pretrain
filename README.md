# OELLM Pretrain Tool

**Scripts to run Megatron-LM sweeps with SLURM job arrays**

This repo provides a wrapper around SLURM for Megatron-LM training, supporting config yaml files and sweeps via job arrays.

- Parse a YAML config with sbatch + Megatron args  
- Expand sweeps and save into `sweep.json`
- Render an sbatch script from a template  
- Submit as a job array  

## Installation

```bash
git clone https://github.com/OpenEuroLLM/oellm_pretrain.git
cd oellm_pretrain
pip install -e .[dev]
```

## Usage

1. Write a config (example: `config/config.yaml`):

```yaml
sbatch_args:
  job_name: oellm_test
  out_dir: /path/to/out

megatron_args:
  lr: [0.0003, 0.0001]
  global_batch_size: 1024
  seq_length: 2048

sweep_args: [lr]
```

2. Launch:
```bash
oellm-pretrain config/example.yaml
```
This will:
- Generate a `sweep.json` file with one entry per sweep point.
- Render and write an sbatch script from the template.
- Submit a SLURM job array, one task per sweep config.
- Create output directories under out_dir/ for logs and checkpoints.

# LUMI

## Installation

```bash
# Set up the environment as we can't (shouldn't) pip install into $HOME or $PYTHONUSERBASE
CONTAINER="/scratch/project_462000963/containers/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-ef203c810cc9.sif"

git clone https://github.com/OpenEuroLLM/oellm_pretrain.git
cd oellm_pretrain
git switch dev_lumi

#TODO: Check this actually works 
singularity exec -B $PWD $CONTAINER bash -c "python3 -m venv venv --system-site-packages; source venv/bin/activate; pip install -e .[dev]"
```

## Usage

1. Write a config (example: `config/config.yaml`):

```yaml
sbatch_args:
  job_name: oellm_test
  out_dir: /path/to/out

megatron_args:
  lr: [0.0003, 0.0001]
  global_batch_size: 1024
  seq_length: 2048

sweep_args: [lr]
```

2. Launch:
```bash
python3 oellm_pretrain/main.py config/example_config_lumi.yaml
```
This will:
- Generate a `sweep.json` file with one entry per sweep point.
- Render and write an sbatch script from the template.
- Submit a SLURM job array, one task per sweep config.
- Create output directories under out_dir/ for logs and checkpoints.