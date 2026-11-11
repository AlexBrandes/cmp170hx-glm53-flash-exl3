# GLM-5.3-Flash on 2× CMP 170HX

This runs GLM-5.3-Flash on two NVIDIA CMP 170HX cards, with the model split
between them. It uses vLLM with turboderp's 2.6-bit EXL3 version of the model and
serves an OpenAI-compatible API that accepts text, images and video. The cards
don't need a fast link between them; a PCIe Gen2 x4 connection each is enough.

One user gets around 75–100 tokens per second, depending on what's being
generated. The full numbers and quality checks are under [Benchmarks](#benchmarks).

```bash
git clone https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3
cd cmp170hx-glm53-flash-exl3
./download.sh     # checkpoint, drafter and INT8 dense weights at pinned revisions (~113 GB)
./start.sh        # sets the power limit (sudo), starts the container, waits for /health
```

The API listens on `127.0.0.1:8000` as model `glm-5.3-flash-exl3-2.60bpw`.
`./stop.sh` stops it. Settings go in `.env` (see `.env.example`).

## Requirements

- 2× NVIDIA CMP 170HX, each exposing 64 GiB. No P2P or link modification needed:
  pipeline parallelism sends one activation per step between the cards.
- Linux x86-64, NVIDIA driver 580 or newer, Docker with the NVIDIA Container Toolkit.
- ~126 GB free on an NVMe SSD while `download.sh` runs (113 GB afterwards: the
  checkpoint, the drafter and the INT8 dense weights), plus the engine image
  (~8 GB) and compilation caches.
- `sudo` for `host-tune.sh` (or `HOST_TUNE=0`; see below).

The tensor-core paths are selected for the measured shapes (74 SMs per card,
this checkpoint); anything else falls back to the scalar EXL3 kernels.

## Benchmarks

Measured 2026-11-11 on 2× CMP 170HX at 170 W per card, using the
[sparkDash](https://github.com/MiaAI-Lab/sparkDash) benchmark protocol at commit
`3e6d4ef`: greedy, thinking off, output forced to 400 tokens, sent to the engine
endpoint. Each cell is the **median of three runs, with the range in brackets**;
sparkDash itself makes one pass per level. The engine serves at most four
sequences, so there is no 8-request level. Every run is in
[results/2026-11-11.json](results/2026-11-11.json).

The measured build is the published image below, built with `docker build --no-cache`
from the pinned sources, serving dense weights that `download.sh` built on the same host.

**Decode, aggregate tok/s**

| Concurrent requests | code | json | prose | structured |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 88.3 (88.1–88.5) | 87.5 (87.1–89.2) | 76.0 (75.7–76.0) | 101.7 (101.3–102.3) |
| 2 | 131.9 (128.2–132.3) | 147.6 (147.5–147.7) | 111.6 (110.9–120.5) | 160.0 (159.7–161.4) |
| 4 | 170.5 (169.2–180.1) | 193.3 (193.1–195.4) | 166.0 (165.4–169.2) | 222.9 (221.7–229.3) |

**Prefill** (one cold request)

| Prompt | Tokens | Prefill tok/s |
| ---: | ---: | ---: |
| 4k | 4,129 | 2,474 (2,448–2,479) |
| 8k | 8,224 | 2,541 (2,541–2,555) |
| 16k | 16,415 | 2,521 (2,508–2,526) |
| 32k | 32,800 | 2,501 (2,497–2,504) |
| 64k | 65,567 | 2,471 (2,469–2,476) |
| 128k | 131,105 | 2,420 (2,420–2,424) |

**Decode at depth** (our addition, not part of sparkDash: one request after a
long prompt, 400 tokens)

| Context | Decode tok/s |
| ---: | ---: |
| 4k | 70.7 (70.5–70.8) |
| 8k | 73.1 (70.4–74.7) |
| 16k | 71.1 (69.4–71.9) |
| 32k | 70.5 (68.3–71.3) |
| 64k | 69.4 (66.8–73.1) |
| 128k | 72.7 (71.7–75.0) |

### Quality

Sent to the engine directly. Greedy with thinking off, except GPQA Diamond.

| Check | Result |
|---|---|
| HumanEval, pass@1 (164 problems) | **141/164** (86.0%) |
| GSM8K (1,319 test questions) | **1,228/1,319** (93.1%) |
| GPQA Diamond (198 questions, thinking on) | **174/198** (87.9%) |
| Image and video input | 7/7: a text control, three single images (4000×3000 to 64×64), two images in one message, two short videos |
| Needle retrieval at 4k, 16k, 32k and 64k | 16/16 |
| 225,280-token window | 225,279 prompt tokens + 1 output token accepted; 225,280 refused |

- **HumanEval** uses EvalPlus's instruct prompt as one chat message (1,024
  tokens) and OpenAI's `check_correctness` program (3 s limit). The Codex
  paper's raw-prompt completion mode scores 124/164 here (133/164 on another
  boot of the same engine: its greedy completions often run to the 512-token
  limit, so this mode moves by several problems between runs).
- **GSM8K** runs vLLM's `tests/evals/gsm8k/gsm8k_eval.py` in its chat mode:
  5-shot, 1,024 tokens, the last integer in the reply. The script's default
  completion mode scores 1,192/1,319.
- **GPQA Diamond** uses OpenAI simple-evals' prompt and answer pattern, with
  thinking on, temperature 1.0 and top_p 0.95 (the checkpoint's
  `generation_config.json`), up to 65,536 tokens per reply, one pass.
- With thinking off, replies about video can open with a short description of
  the frames before the answer.

## What runs

| | |
|---|---|
| Engine | Three pins from [AlexBrandes/vllm](https://github.com/AlexBrandes/vllm), in `versions.env`: upstream vLLM's wheel for [`a0d7356b98`](https://github.com/vllm-project/vllm/commit/a0d7356b98e929c17051bac44f5a086e4117974d); the fork's vLLM branch at [`ea2ef399a6`](https://github.com/AlexBrandes/vllm/commit/ea2ef399a614ea72b65b15804000342f629a850f), Python files over the wheel; and the vllm-cmp170hx plugin at [`0c79bc78fd`](https://github.com/AlexBrandes/vllm/commit/0c79bc78fdc30708f1995d5759828d2a08e39f04) |
| Image | `ghcr.io/alexbrandes/cmp170hx-glm53-flash-exl3@sha256:daa037541ed91b304c5f4168b6dad4820362ef25c1e6ba3193bf5aef562ed076` (tag `engine-ea2ef399a6-0c79bc78fd`): CUDA 13.0 Ubuntu 24.04 base pinned by digest, plus the three pins, each fetched by commit and checked by sha256. `docker/build.sh` rebuilds it (two compiler jobs, about 18 minutes, no GPU needed) |
| Checkpoint | [`turboderp/GLM-5.3-Flash-exl3`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/24781209f2862a939cc21e01d256c943bd1a58a3), branch `sc_2.60bpw`, revision `24781209f2862a939cc21e01d256c943bd1a58a3`: 2.5-bit routed experts, 6-bit dense and attention matrices, LM head and vision tower |
| Dense weights | An INT8 copy of every dense text matrix (attention, MLA projections, indexer, shared experts, the three dense MLPs) and the LM head, built by `download.sh` from the original [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/46a050bec13002ad29ee795c16e61cf4fa3c4a9c) weights at revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` ([tools/convert_dense.py](tools/convert_dense.py)): one scale per output channel, 7.9 GB. The dense matrices are stored in EXL3's rotated input space and run as INT8 × INT8 with per-token activation scales; the LM head runs INT8 weights against FP16 activations. The checkpoint's 6-bit copies of these matrices are not loaded |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) at revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`, BF16, 3 draft tokens per step |
| Parallelism | pipeline parallel 2, tensor parallel 1, layer split 23/22 (`VLLM_PP_LAYER_PARTITION`); the drafter sits on the second card |
| Scheduling | up to 2,048 tokens per step; a new prompt prefills 512 tokens per step beside running decodes; prefill and decode tokens never share a step, because the tensor-core kernels are chosen per phase |
| Cache | KV block size 4,608, 225,280-token window, up to 4 sequences, GPU memory utilization 0.97 |
| Graphs | `FULL_DECODE_ONLY` CUDA graphs |
| Chat template | [config/chat_template.jinja](config/chat_template.jinja): honours `enable_thinking=false` and renders image and video placeholders, which the checkpoint's template drops |

[docker/serve.sh](docker/serve.sh) holds every environment variable and engine
argument; `start.sh` mounts it over the copy in the image, so edits take effect
without a rebuild.

The vLLM branch is upstream plus the fork's commits; the
[plugin](https://github.com/AlexBrandes/vllm/tree/plugin) registers the CMP 170HX
(sm_80) platform and carries the kernels. Among them, this model uses:

1. **Tensor-core EXL3 prefill** for the routed experts and the dense 6-bit
   matrices (`VLLM_EXL3_MOE_PREFILL_TC`, `VLLM_EXL3_DENSE_PREFILL_TC`).
2. **Tensor-core EXL3 decode** for the routed experts and every dense matrix,
   with split-K for the narrow ones (`VLLM_EXL3_MOE_DECODE_TC`,
   `VLLM_EXL3_DENSE_DECODE_TC`, `VLLM_EXL3_DENSE_DECODE_SPLITK`).
3. **Chunked KDA attention prefill on sm_80** (`VLLM_SM80_KDA_PREFILL`).
4. **Pipeline scheduling for these kernels**: decodes spread over the in-flight
   micro-batches, one phase per step (`VLLM_PP_SPREAD_DECODES`,
   `VLLM_PP_HOMOGENEOUS_STEPS`).
5. **INT8 dense matrices from the original weights** (`VLLM_EXL3_DENSE_W8`):
   INT8 tensor-core GEMMs for prefill and decode, and an INT8-weight Marlin
   kernel for the LM head. These replace the tensor-core EXL3 paths for every
   dense matrix; the routed experts stay EXL3.

### Host settings (`host-tune.sh`)

`start.sh` runs it through `sudo`; `HOST_TUNE=0` skips it. It sets the **170 W
power limit** the numbers were measured at on both cards, and nothing else.

### Reasoning

Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}`
to turn it off. The speed tables, HumanEval and GSM8K have thinking off.

## Caveats

- One host: 2× CMP 170HX at PCIe Gen2 x4, Intel Core Ultra 5 225, 93 GB RAM,
  Ubuntu 24.04, driver 610.43.02. Other hosts have not been measured.
- The drafter is licensed **CC BY-NC-ND 4.0**: non-commercial use only. It is
  not in the image; `download.sh` fetches it from its model page.
- The EXL3 extension is compiled at build time, so an image rebuilt with
  `docker/build.sh` has its own identity and extension hash.
- `convert_dense.py` rounds on the CPU it runs on: two machines can disagree in
  a few INT8 values (4,188 of 7.9 billion, each by one, between our two hosts).
- The INT8 dense weights take about 1 GB more per card than the 6-bit copies,
  so the KV cache holds 502,313 tokens: about 2.2 requests at the full
  225,280-token window.
- The decode-at-depth table and the three-run medians are additions to the
  sparkDash protocol; everything else follows it.

## Licence

MIT for this repository ([LICENSE](LICENSE)), except `docker/Dockerfile` and
`docker/serve.sh`, which come from the engine fork under Apache-2.0. The engine is
Apache-2.0 and the model and drafter have their own licences; see
[NOTICE](NOTICE) for every component and credit.
