# GLM-5.3-Flash on 2× CMP 170HX

This runs GLM-5.3-Flash on two 64 GB NVIDIA CMP 170HX cards, with the model split
between them. It uses vLLM and serves an OpenAI-compatible API that accepts
text, images and video. The cards don't need a fast link between them; a PCIe
Gen2 x4 connection each is enough.

The checkpoint is
[`codebrandes/GLM-5.3-Flash-Q4NX-2.6bpw`](https://huggingface.co/codebrandes/GLM-5.3-Flash-Q4NX-2.6bpw),
a 2.6-bit EXL3 quantization. Its speed comes from two changes, and from kernel
work built around them:

- **The Q4N-X codebook** for the routed experts, in place of EXL3's default
  `mul1`. It was developed for these cards: its weights decode with integer
  operations straight into INT8 tensor-core operands.
- **An INT8 copy of the dense weights**, shipped with the checkpoint. The
  attention, shared-expert and dense matrices and the LM head run as INT8
  tensor-core GEMMs instead of 6-bit EXL3.

One user gets around 110–175 tokens per second, depending on what's being
generated. That is 1.5–1.8 times the previous version of this recipe, which
served turboderp's EXL3 checkpoint of the same size, and prompts are processed
1.6–1.9 times as fast. Accuracy is nearly identical to that checkpoint. The
numbers are under [Benchmarks](#benchmarks).

```bash
git clone https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3
cd cmp170hx-glm53-flash-exl3
./download.sh     # checkpoint (with its INT8 dense weights) and drafter at pinned revisions (~118 GB)
./start.sh        # starts the container, waits for /health
```

The API listens on `127.0.0.1:8000` as model `glm-5.3-flash-exl3-2.60bpw`.
`./stop.sh` stops it. Settings go in `.env` (see `.env.example`).

## Requirements

- 2× NVIDIA CMP 170HX, each exposing 64 GB or more. No P2P or link modification needed:
  pipeline parallelism sends one activation per step between the cards.
- Linux x86-64, NVIDIA driver 580 or newer, Docker with the NVIDIA Container Toolkit.
- ~120 GB free on an NVMe SSD while `download.sh` runs (118 GB afterwards: the
  checkpoint with its INT8 dense weights, and the 4-bit drafter), plus the engine
  image (~8 GB) and compilation caches.

The kernels are tuned for this card (sm_80, 74 SMs) and this checkpoint. The
Q4N-X routed experts only run on this engine.

## Benchmarks

Both columns were measured on 2× CMP 170HX at 170 W per card. **Previous** is
the last version of this recipe: turboderp's
[`GLM-5.3-Flash-exl3`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)
`sc_2.60bpw` checkpoint (EXL3 with the `mul1` codebook throughout) on that
version's engine. **This** is the current recipe. Both serve the same INT8 dense
weights. The speed difference comes from the Q4N-X routed experts and the kernel
and scheduler work in this engine together; the accuracy difference comes from
the routed experts.

### Speed

[sparkDash](https://github.com/MiaAI-Lab/sparkDash) protocol at commit
`3e6d4ef`: greedy, thinking off, output forced to 400 tokens, sent to the engine
endpoint. Previous is the median of three runs
([results/2026-11-11.json](results/2026-11-11.json)); this is one run
([results/q4nx-2.6bpw.json](results/q4nx-2.6bpw.json)). The engine serves at
most four sequences, so there is no 8-request level.

**Decode, aggregate tok/s**

| Concurrent requests | Kind | Previous | This |
| ---: | --- | ---: | ---: |
| 1 | code | 88.3 | **161.9** |
| 1 | json | 87.5 | **152.2** |
| 1 | prose | 76.0 | **111.1** |
| 1 | structured | 101.7 | **175.7** |
| 2 | code | 131.9 | **218.1** |
| 2 | json | 147.6 | **232.3** |
| 2 | prose | 111.6 | **199.5** |
| 2 | structured | 160.0 | **274.8** |
| 4 | code | 170.5 | **287.2** |
| 4 | json | 193.3 | **349.6** |
| 4 | prose | 166.0 | **282.3** |
| 4 | structured | 222.9 | **427.6** |

**Prefill** (one cold request)

| Prompt | Tokens | Previous tok/s | This tok/s |
| ---: | ---: | ---: | ---: |
| 4k | 4,127 | 2,474 | **4,078** |
| 8k | 8,226 | 2,541 | **4,480** |
| 16k | 16,419 | 2,521 | **4,569** |
| 32k | 32,801 | 2,501 | **4,605** |
| 64k | 65,565 | 2,471 | **4,605** |
| 128k | 131,104 | 2,420 | **4,446** |

**Decode at depth** (our addition, not part of sparkDash: one request after a
long prompt, 400 tokens)

| Context | Previous tok/s | This tok/s |
| ---: | ---: | ---: |
| 4k | 70.7 | **113.8** |
| 8k | 73.1 | **121.3** |
| 16k | 71.1 | **115.9** |
| 32k | 70.5 | **122.3** |
| 64k | 69.4 | **116.7** |
| 128k | 72.7 | **114.7** |

### Accuracy

Against turboderp's EXL3 checkpoint at the same 2.6 bpw (`mul1` codebook
throughout), both served with the same INT8 dense weights:

| | turboderp EXL3 2.6 bpw | Q4N-X 2.6 bpw |
|---|---:|---:|
| Checkpoint size | 108.9 GB | 108.9 GB |
| KL divergence vs BF16 (malaiwah suite) | 0.1559 | **0.1525** |
| KL divergence vs FP8 (fidelity-v1) | 0.1570 | **0.1511** |
| Perplexity, fidelity-v1 (FP8: 2.441) | 2.747 | **2.727** |

Lower is better. The tests are described below.

**KL divergence against the BF16 model**, on the
[malaiwah GLM-5.3-Flash fidelity suite](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1):
teacher-forced 2,048-token windows of code, encyclopedic, literary and
multilingual text, full-vocabulary KL against BF16 reference hidden states
through the suite's BF16 head. Contexts 0–127 (262,016 positions), paired by
context. Lower is better.

| | Previous | This |
|---|---:|---:|
| Mean KL divergence | 0.1559 | **0.1525** |
| Top-1 agreement | 86.81 % | **87.00 %** |

Paired difference −0.0034, standard error 0.0016.

**KL divergence and perplexity against the FP8 release**, our fidelity-v1 test:
the original FP8 checkpoint served by stock vLLM is the reference, and each
build is scored on the same 857 sequences (336,532 tokens, top-20 logprobs):
the FP8 model's replies to 280 chat prompts (prose, QA, grounded QA, code,
reasoning, multilingual, tool calls), 565 WikiText passages and 12 long WikiText
passages. Ratios are this build over the previous one, with 95 % intervals from
a paired bootstrap over sequences; below 1 means closer to FP8.

| Subset | KL, previous | KL, this | KL ratio | Perplexity ratio |
|---|---:|---:|---:|---:|
| All 857 sequences | 0.1570 | **0.1511** | 0.963 [0.952, 0.973] | 0.9925 [0.9893, 0.9956] |
| WikiText (565) | 0.2104 | **0.1985** | 0.944 [0.930, 0.958] | 0.981 [0.976, 0.986] |
| Long WikiText (12) | 0.1756 | **0.1699** | 0.968 [0.949, 0.982] | 0.994 [0.988, 1.001] |
| Chat replies (280) | **0.0586** | 0.0616 | 1.050 [1.024, 1.078] | 1.008 [1.006, 1.011] |

Perplexity over all 857 sequences: FP8 2.441, previous 2.747, this 2.727.

**Tasks**, sent to the engine directly, greedy with thinking off:

| Check | Previous | This |
|---|---:|---:|
| GSM8K, chat, 5-shot (1,319 questions) | 1,228 (93.1 %) | 1,229 (93.2 %) |
| GSM8K, completion, 5-shot (1,319 questions) | 1,192 (90.4 %) | 1,208 (91.6 %) |
| IFEval, prompt-level strict (541 prompts) | not run | 372 (68.8 %) |
| Image and video input | 7/7 | 7/7 |

- **GSM8K** runs vLLM's `tests/evals/gsm8k/gsm8k_eval.py`: 5-shot, the last
  integer in the reply; chat mode with 1,024 tokens, completion mode with 256.
- **IFEval** uses Google's instruction checkers, 1,280 tokens per reply.
- **Image and video**: a text control, three single images (4000×3000 down to
  64×64), two images in one message, two short videos.
- Each task score is one run; a few questions either way is run-to-run noise.

## What runs

| | |
|---|---|
| Engine | Three pins from [AlexBrandes/vllm](https://github.com/AlexBrandes/vllm), in `versions.env`: upstream vLLM's wheel for [`a0d7356b98`](https://github.com/vllm-project/vllm/commit/a0d7356b98e929c17051bac44f5a086e4117974d); the fork's vLLM branch at [`ea2ef399a6`](https://github.com/AlexBrandes/vllm/commit/ea2ef399a614ea72b65b15804000342f629a850f), Python files over the wheel; and the vllm-cmp170hx plugin at [`ff413b2df1`](https://github.com/AlexBrandes/vllm/commit/ff413b2df19718342e02d9763f161d69afce7b4a) |
| Image | `ghcr.io/alexbrandes/cmp170hx-glm53-flash-exl3@sha256:125662d873c809229ec85ebf18ce76f1f7b9a8a86470edd447880c116af32f8f` (tag `engine-ea2ef399a6-ff413b2df1`): CUDA 13.0 Ubuntu 24.04 base pinned by digest, plus the three pins, each fetched by commit and checked by sha256. `docker/build.sh` rebuilds it (two compiler jobs, about 19 minutes, no GPU needed) |
| Checkpoint | [`codebrandes/GLM-5.3-Flash-Q4NX-2.6bpw`](https://huggingface.co/codebrandes/GLM-5.3-Flash-Q4NX-2.6bpw/tree/799c91c775f3cbd98de292329c0efee2ddc5be10) at revision `799c91c775f3cbd98de292329c0efee2ddc5be10`: routed experts of layers 4–44 in Q4N-X at 2.5 bits, layer 3's experts in EXL3 at 3.5, everything else in EXL3 at 6 (MTP 4). Its model card describes the codebook |
| Dense weights | The checkpoint's `int8/dense.safetensors` (7.9 GB): every dense text matrix (attention, MLA projections, indexer, shared experts, the three dense MLPs) and the LM head in INT8, made from the original weights with one scale per output row. The dense matrices are stored in EXL3's rotated input space and run as INT8 × INT8 with per-token activation scales; the LM head runs INT8 weights against FP16 activations. The checkpoint's 6-bit copies of these matrices are not loaded |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) at revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`, converted by `download.sh` to 4 bits in groups of 128 ([tools/convert_draft.py](tools/convert_draft.py)), 3 draft tokens per step. The drafter only proposes tokens; its precision changes how many are accepted, never the output |
| Parallelism | pipeline parallel 2, tensor parallel 1, layer split 23/22 (`VLLM_PP_LAYER_PARTITION`); the drafter sits on the second card |
| Scheduling | up to 2,048 tokens per step, including while other requests decode; prefill and decode tokens never share a step, because the tensor-core kernels are chosen per phase |
| Cache | KV block size 4,608, 225,280-token window, up to 4 sequences, GPU memory utilization 0.97 |
| Graphs | `FULL_DECODE_ONLY` CUDA graphs |
| Chat template | [config/chat_template.jinja](config/chat_template.jinja): honours `enable_thinking=false` and renders image and video placeholders |

[docker/serve.sh](docker/serve.sh) holds every environment variable and engine
argument; `start.sh` mounts it over the copy in the image, so edits take effect
without a rebuild.

The vLLM branch is upstream plus the fork's commits; the
[plugin](https://github.com/AlexBrandes/vllm/tree/plugin) registers the CMP 170HX
(sm_80) platform and carries the kernels. Among them, this model uses:

1. **Q4N-X routed experts on INT8 tensor cores**: weights decode with integer
   operations straight into INT8 operands, against per-row INT8 activations, in
   prefill and decode (`VLLM_EXL3_Q4N_PREFILL`, `VLLM_EXL3_Q4N_DECODE`), with
   decode calls of up to 64 rows and a wide combine of the expert outputs
   (`VLLM_EXL3_MOE_DECODE_TC_MAX_ROWS`, `VLLM_EXL3_MOE_COMBINE_WIDE`).
2. **INT8 dense matrices** (`VLLM_EXL3_DENSE_W8`, `VLLM_EXL3_DENSE_W8_DECODE`):
   INT8 tensor-core GEMMs for prefill and decode, and an INT8-weight Marlin
   kernel for the LM head.
3. **sm_80 attention and mHC kernels**: chunked KDA prefill, the fused decode
   kernels, and mHC decode and prefill (`VLLM_SM80_KDA_PREFILL`,
   `VLLM_AMPERE_DECODE_KERNELS`, `VLLM_SM80_MHC_DECODE`, `VLLM_SM80_MHC_PREFILL`).
4. **Repeatable sparse attention**: the DSA indexer returns its top-k in sorted
   order with canonical ties, so identical requests give identical results
   (`VLLM_SM80_TOPK_SORTED`, `VLLM_SM80_TOPK_CANONICAL_TIES`).
5. **Pipeline scheduling for these kernels**: decodes spread over the in-flight
   micro-batches, one phase per step (`VLLM_PP_SPREAD_DECODES`,
   `VLLM_PP_HOMOGENEOUS_STEPS`). Each decode-only step takes the streams served
   longest ago, so with several requests decoding none waits behind the others
   while a long prompt is prefilled.
6. **Tensor-core EXL3 kernels** for the tensors that stay in EXL3's `mul1`
   codebook (layer 3's experts).

### Power limit

The recipe leaves the cards' power limit alone. The numbers here were measured
at 170 W per card; to match them, set it yourself
(`sudo nvidia-smi -i <gpu> -pl 170`).

### Reasoning

Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}`
to turn it off. The speed tables and task scores have thinking off.

## Caveats

- One host: 2× CMP 170HX at PCIe Gen2 x4, Intel Core Ultra 5 225, 93 GB RAM,
  Ubuntu 24.04, driver 610.43.02. Other hosts have not been measured.
- This build's numbers were measured on our deployment of the same engine,
  weights and settings. This recipe run end to end (`download.sh`, `start.sh`
  and the pinned image) gave bitwise-identical outputs to that deployment on
  1k–16k-token prompts and passed the same image and video checks.
- The previous column was recorded on that version's engine and was not re-run
  to pair it with this one.
- The drafter is licensed **CC BY-NC-ND 4.0**: non-commercial use only. It is
  not in the image or the checkpoint; `download.sh` fetches it from its model
  page and converts it locally. `download.sh` reports whether the result matches
  the measured build's drafter byte for byte; a different CPU can round a few
  values differently, which changes acceptance slightly, not the output.
- The EXL3 extension is compiled at build time, so an image rebuilt with
  `docker/build.sh` has its own identity and extension hash.
- The KV cache holds 520,579 tokens: about 2.3 requests at the full
  225,280-token window.
- The decode-at-depth table and the three-run medians are additions to the
  sparkDash protocol; everything else follows it.

## Licence

MIT for this repository ([LICENSE](LICENSE)), except `docker/Dockerfile` and
`docker/serve.sh`, which come from the engine fork under Apache-2.0. The engine is
Apache-2.0 and the model and drafter have their own licences; see
[NOTICE](NOTICE) for every component and credit.
