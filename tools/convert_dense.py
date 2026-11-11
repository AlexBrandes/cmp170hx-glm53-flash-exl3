#!/usr/bin/env python3
"""Build the INT8 dense sidecar the engine serves with VLLM_EXL3_DENSE_W8.

For every EXL3 text matrix (layers and lm_head) whose original weight is in <dense.safetensors>:
  W  = original weight [N, K] (BF16, or FP8 e4m3 times its 128x128 block scale), in fp32
  dense matrices: W_r = rows of (W / suh) rotated by a 128-point Hadamard transform along K, i.e.
      EXL3's input space for that matrix (x W^T = H128(x * suh) W_r^T, suh from the EXL3 checkpoint);
      then int8 with one scale per output row: w8 = round(W_r / sw), sw = max|row| / 127
  lm_head: not rotated (served by a Marlin W8A16 kernel)
Writes <prefix>.w8 (int8 [N, K]), <prefix>.sw (fp32 [N]) and <prefix>.suh (fp16 [K]).
Usage: convert_dense.py <dense.safetensors> <exl3 model dir> <out.safetensors>
Runs on CPU in a few minutes. Results can differ from our published file in a few int8 roundings,
depending on the CPU's floating-point summation order.
"""
import json, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, model_dir, outp = sys.argv[1], Path(sys.argv[2]), sys.argv[3]


def hadamard128():
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 128 ** 0.5


H = hadamard128()
orig = safe_open(src, "pt")
okeys = set(orig.keys())
index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
suh_keys = sorted(k for k in index if k.endswith(".suh") and (
    k == "lm_head.suh" or (k.startswith("model.language_model.layers.") and ".experts." not in k)))
handles = {}


def exl3_tensor(key):
    f = index[key]
    if f not in handles:
        handles[f] = safe_open(str(model_dir / f), "pt")
    return handles[f].get_tensor(key)


def load(name):
    w = orig.get_tensor(name + ".weight")
    if w.dtype == torch.float8_e4m3fn:
        si = orig.get_tensor(name + ".weight_scale_inv").float()
        n, k = w.shape
        w = (w.float().reshape(n // 128, 128, k // 128, 128) * si[:, None, :, None]).reshape(n, k)
    return w.float()


def originals(prefix):
    names = ([prefix[:-len("qkv_proj")] + p for p in ("q_proj", "k_proj", "v_proj")]
             if prefix.endswith(".self_attn.qkv_proj") else [prefix])
    if not all(n + ".weight" in okeys for n in names):
        return None
    return torch.cat([load(n) for n in names], 0)


out, t0, done = {}, time.time(), 0
for key in suh_keys:
    prefix = key[:-len(".suh")]
    W = originals(prefix)
    if W is None:
        continue
    suh = exl3_tensor(key).float()
    n, k = W.shape
    if k != suh.numel():
        continue
    rotate = prefix != "lm_head"
    Wr = ((W / suh).reshape(n, -1, 128) @ H).reshape(n, k) if rotate else W
    sw = Wr.abs().amax(1).clamp_min(1e-12) / 127
    out[prefix + ".w8"] = (Wr / sw[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
    out[prefix + ".sw"] = sw.contiguous()
    if rotate:
        out[prefix + ".suh"] = suh.half().contiguous()
    done += 1
    if done % 20 == 0:
        print(f"{done} matrices, {time.time() - t0:.0f}s", flush=True)
save_file(out, outp, metadata={"format": "glm53-dense-w8-v1", "source": "zai-org/GLM-5.3-Flash"})
print(f"wrote {outp}: {done} matrices")
