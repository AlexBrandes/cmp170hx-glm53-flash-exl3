#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GLM-5.3-Flash EXL3 2.6 bpw with Q4NX routed experts, PP2 / TP1, DFlash2 k3, on two CMP 170HX cards.
# /model: codebrandes/GLM-5.3-Flash-Q4NX-2.6bpw (includes the INT8 dense weights in int8/);
# /draft: the 4-bit drafter download.sh builds from incoai/GLM-5.3-Flash-DFlash2.
set -euo pipefail

export VLLM_CACHE_ROOT=/cache/vllm
export TRITON_CACHE_DIR=/cache/triton
export CUDA_CACHE_PATH=/cache/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pipeline split: 23 target layers on the first card, 22 and the drafter on the second.
export VLLM_PP_LAYER_PARTITION=23,22
# Scheduling: spread decodes over the in-flight micro-batches (each decode-only
# step takes the streams served longest ago, so none waits behind the others),
# and never mix prefill and decode tokens in one step (the tensor-core paths
# are per phase).
export VLLM_PP_SPREAD_DECODES=1
export VLLM_PP_HOMOGENEOUS_STEPS=1
# sm_80 attention and mHC kernels: KDA chunked prefill, the fused decode family,
# mHC decode and the mHC prefill pre-norm.
export VLLM_SM80_KDA_PREFILL=1
export VLLM_AMPERE_DECODE_KERNELS=1
export VLLM_SM80_MHC_DECODE=1
export VLLM_SM80_MHC_PREFILL=1
# DSA indexer top-k in sorted order with canonical ties: identical requests give
# identical results (otherwise prompts over 2,048 tokens vary run to run).
export VLLM_SM80_TOPK_SORTED=1
export VLLM_SM80_TOPK_CANONICAL_TIES=1
# Routed experts (Q4NX): INT8 tensor-core GEMMs with per-row INT8 activations in
# prefill and decode, decode calls up to 64 rows, and the wide combine.
export VLLM_EXL3_MOE_PREFILL_TC=1
export VLLM_EXL3_MOE_DECODE_TC=1
export VLLM_EXL3_MOE_DECODE_TC_MODE=full
export VLLM_EXL3_Q4N_PREFILL=i8
export VLLM_EXL3_Q4N_I8_GEMM=t2c4
export VLLM_EXL3_Q4N_DECODE=i8
export VLLM_EXL3_Q4N_DECODE_GEMM=t2c4
export VLLM_EXL3_MOE_DECODE_TC_MAX_ROWS=64
export VLLM_EXL3_MOE_COMBINE_WIDE=1
export VLLM_EXL3_ROUTE_CHECK=0
# EXL3 tensor-core paths for the layer-3 experts and any dense matrix not served
# from the INT8 file.
export VLLM_EXL3_DENSE_PREFILL_TC=1
export VLLM_EXL3_DENSE_PREFILL_ONE_PRODUCT=float
export VLLM_EXL3_DENSE_DECODE_TC=1
export VLLM_EXL3_DENSE_DECODE_TC_FAMILIES=all
export VLLM_EXL3_DENSE_DECODE_SPLITK=1
export VLLM_EXL3_DECODE_VARIANT=off
# Dense matrices and LM head from the checkpoint's INT8 copy instead of its 6-bit
# EXL3 tensors: INT8 tensor-core GEMMs for prefill and decode, Marlin W8A16 for
# the LM head. Unset both to serve them from EXL3.
export VLLM_EXL3_DENSE_W8=/model/int8/dense.safetensors
export VLLM_EXL3_DENSE_W8_DECODE=1

exec vllm serve /model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name glm-5.3-flash-exl3-2.60bpw \
  --quantization exl3 --load-format exl3 --dtype bfloat16 \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 --mm-encoder-tp-mode data \
  --max-model-len 225280 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --prefill-chunk-with-decodes 2048 --block-size 4608 \
  --gpu-memory-utilization 0.97 \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
  --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"method": "dflash", "model": "/draft", "num_speculative_tokens": 3, "quantization": null, "draft_load_config": {"load_format": "safetensors"}}' \
  --hf-overrides '{"text_config": {"aux_hidden_state_tensor": "stream_mean"}}' \
  "$@"
