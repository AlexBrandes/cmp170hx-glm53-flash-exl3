# GLM-5.3-Flash on 2× CMP 170HX

**GLM-5.3-Flash (EXL3, 2.60 bpw) on two NVIDIA CMP 170HX cards, pipeline-parallel
with vLLM: 83 tok/s for one user (structured), 192 tok/s across four, about 1,800
tok/s prefill from 4k to 128k prompts, a 225,280-token window, image and video
input, and HumanEval 142/164, GSM8K 1,228/1,319, GPQA Diamond 172/198.** Measured
on PCIe Gen2 x4 links.

Serves turboderp's [EXL3 quantization of GLM-5.3-Flash](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)
with upstream vLLM and the [vllm-cmp170hx](https://github.com/AlexBrandes/vllm)
plugin in a container: an OpenAI-compatible API with text, image and video
input, DFlash speculative decoding, reasoning and tool calls. The plugin carries
tensor-core EXL3 kernels for these cards (sm_80) for both prefill and decode;
everything runs on the two GPUs.

```bash
git clone https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3
cd cmp170hx-glm53-flash-exl3
./download.sh     # checkpoint and drafter at the pinned revisions (~105 GB)
./start.sh        # sets the power limit (sudo), starts the container, waits for /health
```

The API listens on `127.0.0.1:8000` as model `glm-5.3-flash-exl3-2.60bpw`.
`./stop.sh` stops it. Settings go in `.env` (see `.env.example`).

## Requirements

- 2× NVIDIA CMP 170HX, each exposing 64 GiB. No P2P or link modification needed:
  pipeline parallelism sends one activation per step between the cards.
- Linux x86-64, NVIDIA driver 580 or newer, Docker with the NVIDIA Container Toolkit.
- ~110 GB free on an NVMe SSD for the checkpoint and drafter, plus the engine
  image (~26 GB) and compilation caches.
- `sudo` for `host-tune.sh` (or `HOST_TUNE=0`; see below).

The tensor-core paths are selected for the measured shapes (74 SMs per card,
this checkpoint); anything else falls back to the scalar EXL3 kernels.

## Benchmarks

Measured 2026-11-05 on 2× CMP 170HX at 170 W per card, using the
[sparkDash](https://github.com/MiaAI-Lab/sparkDash) benchmark protocol at commit
`3e6d4ef`: greedy, thinking off, output forced to 400 tokens, sent to the engine
endpoint. Each cell is the **median of three runs, with the range in brackets**;
sparkDash itself makes one pass per level. The engine serves at most four
sequences, so there is no 8-request level. Every run is in
[results/2026-11-05.json](results/2026-11-05.json).

The measured build is plugin `947ef7c7e1` on vLLM branch `2a4e137909`. The
published image adds plugin `d5cb84150b`, which fixes an engine crash when a
long prompt arrives while other requests are decoding and changes nothing else
this model runs; the compiled EXL3 extension is byte-identical.

**Decode, aggregate tok/s**

| Concurrent requests | code | json | prose | structured |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 76.5 (76.3–76.5) | 70.9 (70.4–71.4) | 57.8 (57.8–58.0) | 83.0 (82.9–83.1) |
| 2 | 109.9 (109.4–110.0) | 113.2 (112.5–118.3) | 92.4 (92.2–92.7) | 122.5 (122.4–122.8) |
| 4 | 151.4 (144.6–153.4) | 164.6 (163.7–166.8) | 139.6 (137.7–141.2) | 191.9 (178.2–202.0) |

**Prefill** (one cold request)

| Prompt | Tokens | Prefill tok/s |
| ---: | ---: | ---: |
| 4k | 4,126 | 1,741 (1,741–1,747) |
| 8k | 8,222 | 1,852 (1,851–1,854) |
| 16k | 16,416 | 1,842 (1,839–1,846) |
| 32k | 32,798 | 1,836 (1,832–1,839) |
| 64k | 65,566 | 1,823 (1,820–1,826) |
| 128k | 131,103 | 1,803 (1,803–1,806) |

**Decode at depth** (our addition, not part of sparkDash: one request after a
long prompt, 400 tokens)

| Context | Decode tok/s |
| ---: | ---: |
| 4k | 59.1 (55.6–59.3) |
| 8k | 59.0 (55.4–59.4) |
| 16k | 60.8 (60.0–61.0) |
| 32k | 58.1 (55.9–58.8) |
| 64k | 56.1 (54.9–58.0) |
| 128k | 59.0 (58.8–59.5) |

### Quality

Sent to the engine directly. Greedy with thinking off, except GPQA Diamond.

| Check | Result |
|---|---|
| HumanEval, pass@1 (164 problems) | **142/164** (86.6%) |
| GSM8K (1,319 test questions) | **1,228/1,319** (93.1%) |
| GPQA Diamond (198 questions, thinking on) | **172/198** (86.9%) |
| Image and video input | 7/7: a text control, three single images (4000×3000 to 64×64), two images in one message, two short videos |
| Needle retrieval at 4k, 16k, 32k and 64k | 16/16 |
| 225,280-token window | 225,279 prompt tokens + 1 output token accepted; 225,280 refused |

- **HumanEval** uses EvalPlus's instruct prompt as one chat message (1,024
  tokens) and OpenAI's `check_correctness` program (3 s limit). The Codex
  paper's raw-prompt completion mode scores 134/164.
- **GSM8K** runs vLLM's `tests/evals/gsm8k/gsm8k_eval.py` in its chat mode:
  5-shot, 1,024 tokens, the last integer in the reply. The script's default
  completion mode scores 1,190/1,319.
- **GPQA Diamond** uses OpenAI simple-evals' prompt and answer pattern, with
  thinking on, temperature 1.0 and top_p 0.95 (the checkpoint's
  `generation_config.json`), up to 65,536 tokens per reply, one pass.
- With thinking off, replies about video can open with a short description of
  the frames before the answer.

## What runs

| | |
|---|---|
| Engine | Three pins from [AlexBrandes/vllm](https://github.com/AlexBrandes/vllm), in `versions.env`: upstream vLLM's wheel for [`a0d7356b98`](https://github.com/vllm-project/vllm/commit/a0d7356b98e929c17051bac44f5a086e4117974d); the fork's vLLM branch at [`ea2ef399a6`](https://github.com/AlexBrandes/vllm/commit/ea2ef399a614ea72b65b15804000342f629a850f), Python files over the wheel; and the vllm-cmp170hx plugin at [`d5cb84150b`](https://github.com/AlexBrandes/vllm/commit/d5cb84150bb0327224d7caa72aae1135f2bd2d92) |
| Image | `ghcr.io/alexbrandes/cmp170hx-glm53-flash-exl3@sha256:70787b9e70ac9ddee24618cd491a19a27da1cc55e3fd842492900bbbb0bd9488` (tag `engine-ea2ef399a6-d5cb84150b`): CUDA 13.0 Ubuntu 24.04 base pinned by digest, plus the three pins, each fetched by commit and checked by sha256. `docker/build.sh` rebuilds it (two compiler jobs, about 17 minutes, no GPU needed) |
| Checkpoint | [`turboderp/GLM-5.3-Flash-exl3`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/24781209f2862a939cc21e01d256c943bd1a58a3), branch `sc_2.60bpw`, revision `24781209f2862a939cc21e01d256c943bd1a58a3`: 2.5-bit routed experts, 6-bit dense and attention matrices, LM head and vision tower |
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
- The decode-at-depth table and the three-run medians are additions to the
  sparkDash protocol; everything else follows it.

## Licence

MIT for this repository ([LICENSE](LICENSE)), except `docker/Dockerfile` and
`docker/serve.sh`, which come from the engine fork under Apache-2.0. The engine is
Apache-2.0 and the model and drafter have their own licences; see
[NOTICE](NOTICE) for every component and credit.
