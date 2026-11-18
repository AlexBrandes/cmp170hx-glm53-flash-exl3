"""Quantize the DFlash2 drafter (BF16) to 4 bits for vLLM (compressed-tensors, pack-quantized).

Every weight of a module that vLLM builds with the draft's quant_config (self_attn q/k/v/o, mlp gate/up/down,
fc, attention_conv / mlp_conv kernel_projection) becomes symmetric 4-bit round-to-nearest in groups of 128 along
K, with a per-group clip chosen from {1.0 .. 0.75} by squared error; vLLM serves it with Marlin W4A16. Stored as
weight_packed (int32 [N, K / 8], low bits first along K, value + 8), weight_scale (bf16 [N, K / 128]) and
weight_shape (int64 [2]). Everything else is copied. The drafter only proposes tokens, so its quantization changes
how many are accepted, never the verified output.
Usage: python convert_draft.py <bf16 drafter dir> <out dir>
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1:3]
mode = "w4"
BITS, GROUP = 4, 128
CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75) if BITS == 4 else (1.0,)
QMAX = 2 ** (BITS - 1) - 1  # 7 / 127
QMIN = -(2 ** (BITS - 1))   # -8 / -128
QUANT = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
         ".self_attn.o_proj.weight", ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
         ".kernel_projection.weight")


def wanted(name):
    return name == "fc.weight" or name.endswith(QUANT)


def quantise(w):
    n, k = w.shape
    g = GROUP or k
    x = w.float().view(n, k // g, g)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_q, best_s, best_e = None, None, None
    for c in CLIPS:
        s = c * amax / QMAX
        q = (x / s).round().clamp(QMIN, QMAX)
        e = (q * s - x).pow(2).sum(-1, keepdim=True)
        if best_q is None:
            best_q, best_s, best_e = q, s, e
        else:
            m = e < best_e
            best_q, best_s, best_e = torch.where(m, q, best_q), torch.where(m, s, best_s), torch.where(m, e, best_e)
    q = best_q.view(n, k)
    s = best_s.view(n, k // g)
    err = float(((q.view(n, k // g, g) * best_s).view(n, k) - w.float()).norm() / w.float().norm())
    return q, s, err


def pack(q):
    n, k = q.shape
    per = 32 // BITS
    u = (q.to(torch.int64) - QMIN).view(n, k // per, per)
    out = torch.zeros(n, k // per, dtype=torch.int64)
    for i in range(per):
        out |= u[:, :, i] << (BITS * i)
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32).contiguous()


os.makedirs(dst, exist_ok=True)
f = safe_open(os.path.join(src, "model.safetensors"), "pt")
out, report = {}, {}
for name in f.keys():
    t = f.get_tensor(name)
    if wanted(name):
        q, s, err = quantise(t)
        base = name[: -len(".weight")]
        out[base + ".weight_packed"] = pack(q)
        out[base + ".weight_scale"] = s.to(torch.bfloat16).contiguous()
        out[base + ".weight_shape"] = torch.tensor(list(t.shape), dtype=torch.int64)
        report[base] = {"shape": list(t.shape), "rel_l2": round(err, 5)}
        print(base, list(t.shape), round(err, 5), flush=True)
    else:
        out[name] = t.contiguous()
save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors", "format": "pack-quantized", "quantization_status": "compressed",
    "ignore": ["candidate_selector.hidden_projection"],
    "config_groups": {"group_0": {"targets": ["Linear"], "input_activations": None, "output_activations": None,
                                  "weights": {"num_bits": BITS, "type": "int", "symmetric": True, "dynamic": False,
                                              "strategy": "group" if GROUP else "channel", "group_size": GROUP,
                                              "actorder": None, "observer": "minmax"}}},
}
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
for extra in ("MODEL_CARD.txt", "download-receipt.json"):
    if os.path.exists(os.path.join(src, extra)):
        shutil.copy(os.path.join(src, extra), dst)
json.dump(report, open(os.path.join(dst, "quant-report.json"), "w"), indent=1)
print("DONE", mode, len(report), "matrices; mean rel_l2", round(sum(r["rel_l2"] for r in report.values()) / len(report), 5))
