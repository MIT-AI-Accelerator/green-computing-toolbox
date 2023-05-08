#!/bin/bash
# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

NVPOWERCMD="/usr/bin/nvidia-smi -pl "
DEFAULT_POWER_LIMIT=250

if [ ! -z "$SLURM_JOB_GPUS" ]; then
    SLURM_GPUVAR=$SLURM_JOB_GPUS
elif [ ! -z "$SLURM_STEP_GPUS" ]; then
    SLURM_GPUVAR=$SLURM_STEP_GPUS
elif [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    SLURM_GPUVAR=$CUDA_VISIBLE_DEVICES
else
    exit 0
fi

GPUS=${SLURM_GPUVAR//,/ }
for ${CUDA_DEVNUM} in ${GPUS}
do
    if [ ! -z "$SPANK_GPUPOWER" ]; then
	${NVPOWERCMD} ${SPANK_GPUPOWER} -i ${CUDA_DEVNUM}
    else
        ${NVPOWERCMD} ${DEFAULT_POWER_LIMIT} -i ${CUDA_DEVNUM}
    fi
done

