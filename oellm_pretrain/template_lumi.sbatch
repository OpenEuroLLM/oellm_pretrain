#!/bin/bash

#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --threads-per-core=1
#SBATCH --output=$slurm_logs_dir/%x-%A-%a.out
#SBATCH --error=$slurm_logs_dir/%x-%A-%a.err
#SBATCH --job-name=$job_name

######################################################################
# PATHS
#
# {out_dir}/
# ├── slurm_output/
# │   ├── %n-%j-%a-{TIMESTAMP}.err
# │   └── %n-%j-%a-{TIMESTAMP}.out
# └── checkpoints/
#     └── %n/
#         ├── tensorboard/
#         ├── wandb/
#         ├── iter_0001000/
#         ├── iter_0002000/
#         └── ...
#
# {MEGATRON_CACHE_BASE}/
# └── {USER}/
#     ├── MEGATRON_CACHEDIR/
#     ├── APPTAINER_CACHEDIR/
#     └── APPTAINER_TMPDIR/
#
######################################################################

######################################################################
# PATHS and ENV VARS
######################################################################

# Bash "strict mode" (see http://redsymbol.net/articles/unofficial-bash-strict-mode/)
set -euo pipefail

# When slurm reschedules a job that ended on node failure, it will run
# with the same job ID, clobbering the original logs. Rename the logs
# and include timestamp to avoid this.
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
logid="${SLURM_JOB_NAME}-${SLURM_JOBID}-${SLURM_ARRAY_TASK_ID}"
[[ -e "$slurm_logs_dir/$logid.out" ]] && mv -f "$slurm_logs_dir/$logid.out" "$slurm_logs_dir/$logid-$timestamp.out"
[[ -e "$slurm_logs_dir/$logid.err" ]] && mv -f "$slurm_logs_dir/$logid.err" "$slurm_logs_dir/$logid-$timestamp.err"

# SOURCE CLUSTER SETUP SCRIPT
SETUP_SCRIPT=$cluster_setup_script
chmod +x $SETUP_SCRIPT
source $SETUP_SCRIPT

# From sweep.json, extract the row for this task, then get jobname and args
ROW=$(jq -c ".[$SLURM_ARRAY_TASK_ID]" "$sweep_path")
JOBNAME=$(jq -r '.job_name' <<<"$ROW")
mapfile -t ARGS < <(jq -r '.flags[]' <<<"$ROW")

# OUTPUT_DIR
CHECKPOINT_PATH="$out_dir/checkpoints/$JOBNAME"
TENSORBOARD_DIR="${CHECKPOINT_PATH}/tensorboard"
WANDB_DIR="${CHECKPOINT_PATH}/wandb"
mkdir -p "$out_dir" "$CHECKPOINT_PATH" "$TENSORBOARD_DIR" "$WANDB_DIR"

######################################################################
# PARSE ARGS FROM YAML CONFIG
######################################################################

# Append args depending on path
ARGS+=(
    --load "${CHECKPOINT_PATH}"
    --save "${CHECKPOINT_PATH}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --wandb-save-dir "${WANDB_DIR}"
    --data-cache-path "${MEGATRON_CACHE}"
)

######################################################################
# RUN
######################################################################

CMD="${MEGATRON_PATH}/pretrain_gpt.py ${ARGS[*]}"

echo '=============== CMD: ==============='
echo $CMD
echo '===================================='

cd $MEGATRON_PATH

srun \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOBID \
    bash -c \
    "singularity exec  --bind $BIND_DIRS $CONTAINER \
    python -u -m torch.distributed.run \
      --nproc-per-node $SLURM_GPUS_ON_NODE \
      --nnodes $SLURM_NNODES \
      --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      --rdzv_backend c10d \
      --max_restarts 0 \
      --tee 3 \
      --node_rank \$SLURM_PROCID \
      --role \$(hostname -s|tr -dc '0-9'): \
      $CMD"

echo "END $SLURM_JOBID: $(date)"
