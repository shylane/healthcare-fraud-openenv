#!/bin/bash
export UNSLOTH_DISABLE_STATISTICS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /root/workspace
exec .venv/bin/python training/orchestrate_cycles_v2.py   --workspace /root/workspace   --model-id unsloth/Qwen3-1.7B   --total-cycles 3   --train-episodes 15   --group-size 32   --max-completion-length 1024   --max-seq-length 4096   --max-steps-per-cycle 1500   --lora-rank 32   --num-iterations 2   --lr 5e-6
