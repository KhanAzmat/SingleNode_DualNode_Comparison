#!/usr/bin/env bash
set -euo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate dpsk14b
export RAY_OVERRIDE_NODE_IP=10.200.0.2
export VLLM_HOST_IP=10.200.0.2
export HOST_IP=10.200.0.2
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export NCCL_SOCKET_NTHREADS=2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=4194304
export NCCL_ALGO=Ring
export NCCL_TREE_THRESHOLD=0
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_LAUNCH_MODE=PARALLEL

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
exec ray start --address=10.200.0.1:6379 --node-ip-address=10.200.0.2







