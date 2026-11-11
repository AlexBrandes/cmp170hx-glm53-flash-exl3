"""Fetch the dense text weights of the original GLM-5.3-Flash checkpoint (zai-org/GLM-5.3-Flash at a
pinned revision) into one safetensors file: attention q/k/v/o, MLA q_a/q_b/kv_a/o, indexer wq_b,
shared experts, the three dense MLPs and lm_head, with their FP8 block scales (13.2 GB of the ~600 GB
checkpoint). Uses HTTP range requests on the safetensors shards, so the routed experts are never
downloaded. Usage: fetch_dense.py <repo> <revision> <out.safetensors>"""
import json, re, struct, sys, urllib.request
from pathlib import Path
REPO = f'https://huggingface.co/{sys.argv[1]}/resolve/{sys.argv[2]}/'
OUT = Path(sys.argv[3])
def get(url, rng=None, tries=5):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}'} if rng else {})
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if t == tries - 1: raise
            print('retry', url[-40:], e, flush=True)
idx = json.loads(get(REPO + 'model.safetensors.index.json'))['weight_map']
pat = re.compile(r'^model\.language_model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|indexer\.wq_b)|mlp\.shared_experts\.(gate|up|down)_proj|mlp\.(gate|up|down)_proj)\.(weight|weight_scale_inv)$')
want = sorted(k for k in idx if (pat.match(k) and int(pat.match(k).group(1)) <= 44) or k in ('lm_head.weight',))
headers = {}
meta, off = {}, 0
for k in want:
    shard = idx[k]
    if shard not in headers:
        n = struct.unpack('<Q', get(REPO + shard, (0, 7)))[0]
        headers[shard] = (n, json.loads(get(REPO + shard, (8, 8 + n - 1))))
    n, h = headers[shard]; m = h[k]; a, b = m['data_offsets']
    meta[k] = {'dtype': m['dtype'], 'shape': m['shape'], 'data_offsets': [off, off + b - a], '_src': (shard, 8 + n + a, 8 + n + b - 1)}
    off += b - a
print(len(want), 'tensors', round(off / 1e9, 2), 'GB', flush=True)
hdr = json.dumps({k: {x: v[x] for x in ('dtype', 'shape', 'data_offsets')} for k, v in meta.items()}).encode()
hdr += b' ' * (-len(hdr) % 8)
with open(OUT, 'wb') as f:
    f.write(struct.pack('<Q', len(hdr))); f.write(hdr)
    done = 0
    for i, k in enumerate(want):
        shard, a, b = meta[k]['_src']
        CH = 256 << 20
        for s in range(a, b + 1, CH):
            f.write(get(REPO + shard, (s, min(b, s + CH - 1))))
        done += b - a + 1
        if i % 20 == 0: print(f'{i}/{len(want)} {done/1e9:.2f} GB', flush=True)
print('DONE', OUT, off, flush=True)
