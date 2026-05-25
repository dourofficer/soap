#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
RUN="python -m experiments.reps.run"
CFG="--config experiments/reps/configs/hidden.yaml"

# $RUN grads  $CFG --set grads.loss=kl_temp    
# $RUN grads  $CFG --set grads.loss=ntp        
# $RUN grads  $CFG --set grads.loss=kl_uniform 
$RUN hidden $CFG                             