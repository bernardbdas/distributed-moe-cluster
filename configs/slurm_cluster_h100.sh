#!/usr/bin/env bash
#SBATCH --job-name=moe-8x7b-pretrain
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=960G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-h100
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Distributed Environment Setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "=========================================================="
echo "Starting Distributed MoE Pretraining on $SLURM_JOB_NUM_NODES Nodes ($WORLD_SIZE GPUs)"
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "=========================================================="

srun --kill-on-bad-exit=1 \
     python3 -m torch.distributed.run \
     --nproc_per_node=8 \
     --nnodes=$SLURM_JOB_NUM_NODES \
     --node_rank=$SLURM_NODEID \
     --master_addr=$MASTER_ADDR \
     --master_port=$MASTER_PORT \
     src/cli.py train \
     --config configs/moe_model_8x7b.yaml \
     --strategy fsdp
