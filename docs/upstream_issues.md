# Upstream Library Issues

Issues discovered during healthcare-fraud GSPO training setup (March 2026).
These are suitable for filing or adding to existing GitHub issues.

---

## 1. unsloth — RoPE shape mismatch in fast inference (KV-cache mode)

**Library**: `unsloth` (latest as of March 2026)
**File**: `unsloth/models/llama.py`, function `LlamaAttention_fast_forward_inference`, ~line 506
**Affects**: Any GRPO/PPO training that uses `generate()` with KV-cache on Qwen2/Llama-family models

### Symptom
```
RuntimeError: output with shape [96, 12, 1, 128] doesn't match the broadcast shape [96, 12, 1380, 128]
```
Occurs during GRPO generation when the model produces its first completion token after the
full prompt has been encoded.

### Root cause
During single-token KV-cache inference, `Qn`/`Kn` have shape `[bsz, n_heads, 1, head_dim]`
(one new token). However, `position_ids` is forwarded with the full prompt length
(e.g. 1380 tokens) from the generation loop. The RoPE lookup:
```python
cos = cos[position_ids].unsqueeze(1)  # → [bsz, 1, 1380, head_dim]
Qn *= cos                             # broadcast → [bsz, n_heads, 1380, head_dim] ≠ Qn
```
collapses to an incompatible shape.

### Minimal fix (applied locally)
After `position_ids = position_ids.to(Qn.device)`, insert:
```python
# Fast inference path: Qn/Kn are single-token; slice position_ids to current token only
if position_ids.shape[-1] > 1:
    position_ids = position_ids[:, -1:]
```

### Notes
- Does NOT affect full-sequence (prefill) forward passes — only triggered on the
  single-token generation steps inside KV-cache mode.
- Reproducible with `unsloth/Qwen2.5-1.5B-Instruct` and `unsloth/Qwen3-1.7B` +
  GRPO probe (20 steps, batch 4).
- The same bug exists in `unsloth/models/qwen3.py` (`Qwen3Attention_fast_forward_inference`
  ~line 305), same fix applies.
- Related unsloth issue: likely the same root cause as fast-inference shape reports
  filed against `Mistral` and `Qwen2` in the unsloth repo.

---

## 2. unsloth — dtype mismatch in LoRA kernels due to PyTorch 2.4 AMP API change

**Library**: `unsloth` (latest as of March 2026)
**Files**: `unsloth/kernels/utils.py` (decorator definition), `unsloth/kernels/fast_lora.py`
  (forward/backward of `LoRA_MLP` and `LoRA_W`)
**Affects**: LoRA fine-tuning with `load_in_4bit=True` + `bf16=True` on PyTorch >= 2.4.

### Symptoms
Forward:
```
RuntimeError: self and mat2 must have the same dtype, but got Half and Float
```
Backward:
```
RuntimeError: self and mat2 must have the same dtype, but got Float and Half
```
Triggered in `LoRA_MLP.forward`/`backward` and related kernels.

### Root cause (PyTorch 2.4 API regression)
`unsloth` relies on `@torch.cuda.amp.custom_fwd` / `@torch.cuda.amp.custom_bwd`
(deprecated in PyTorch 2.4). The old API **automatically cast float inputs to the
autocast dtype** (float16) inside the forward, ensuring `X.dtype == float16` for all
saved activations.

The new API `torch.amp.custom_fwd(device_type='cuda')` with `cast_inputs=None` (the
default) does **not** cast inputs. So `X` stays float32 in the forward/backward, but
all matmul outputs inside the autocast context are float16. This creates a cascading
dtype mismatch:

```python
# In utils.py — these decorators now have different semantics:
torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")  # BUG: no cast_inputs

# In fast_lora.py forward:
dtype = X.dtype         # = float32 (not cast by new decorator)
out = torch_matmul(X, W.t())   # autocast → float16
XA = X @ A.to(dtype)           # float32 @ float32 → float16 (autocast!)
out.addmm_(XA, B.to(dtype))    # FAIL: out=float16, mat2=float32

# In fast_lora.py backward (LoRA_MLP):
dtype = X.dtype         # = float32 (X saved as float32)
d_downA = empty_like(downA.to(dtype))  # → float32
h = matmul_lora(dY, ...)       # → float16 (autocast)
dY @ downB.t()                 # float32 @ float32 → float16 (autocast!)
d_downA.addmm_(h.t(), dY @ downB.t())  # FAIL: float32.addmm_(float16, float16)
```

### Fix (applied locally to `utils.py`)
```python
# Restore old cast_inputs behaviour explicitly:
torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
```
This ensures X is cast to float16 before the forward body runs (and saved as float16),
making all downstream dtype assumptions consistent — exactly as the old API behaved.

### Notes
- Reproducible with `unsloth/Qwen2.5-1.5B-Instruct` + `load_in_4bit=True` + `bf16=True`
  GRPO training on PyTorch 2.11.0+cu128.
- The same root cause produces TWO different error messages depending on which
  addmm_ operation is hit first (forward vs backward).
- Upstream fix needed in `utils.py` at the `torch_amp_custom_fwd` definition site.

---

## 3. unsloth — `get_statistics()` crashes on DNS failure at model load time

**Library**: `unsloth` (latest as of March 2026)
**File**: `unsloth/_utils.py` (or `unsloth/utils.py`), `get_statistics()` function
**Affects**: Any environment with intermittent/no outbound DNS (cloud VMs, firewalled
training nodes).

### Symptom
```
socket.gaierror: [Errno -3] Temporary failure in name resolution
```
Raised during `FastLanguageModel.from_pretrained(...)`, not from HuggingFace model
download but from an internal telemetry call that uses `snapshot_download()`.

### Root cause
`get_statistics()` calls `huggingface_hub.snapshot_download()` to report usage stats
back to Unsloth. If DNS resolves transiently (common on Vast.ai), this raises an
uncaught exception that kills the entire model load.

### Workaround (environment variable)
```bash
export UNSLOTH_DISABLE_STATISTICS=1
```
This env var disables the telemetry call entirely.

### Suggested fix for upstream
Wrap the `snapshot_download()` call in a `try/except (OSError, socket.gaierror, Exception):`
and silently swallow failures, since telemetry is non-critical.

---

## 4. unsloth / unsloth-zoo — packaging constraint `trl <= 0.24.0` is too strict

**Library**: `unsloth-zoo` (pulled as dependency of `unsloth`)
**File**: `unsloth_zoo/pyproject.toml` (or `setup.py`), dependency on `trl`
**Affects**: Anyone who needs `trl >= 0.29` (e.g. for `GRPOConfig.importance_sampling_level`).

### Symptom
`uv pip install unsloth trl==0.29.1` fails with a dependency conflict:
```
unsloth-zoo requires trl<=0.24.0
```

### Root cause
`unsloth-zoo` pins an old upper bound on `trl` to stay compatible with the internal
`UnslothGRPOTrainer` code, but the actual runtime code generated by unsloth at
`/workspace/unsloth_compiled_cache/UnslothGRPOTrainer.py` works fine with trl 0.29.1
because it calls `trl.GRPOTrainer` directly. The constraint is a packaging artifact
that doesn't reflect actual runtime compatibility.

### Workaround
Two-pass `uv` install:
```bash
# Pass 1: let unsloth pick its preferred trl (≤ 0.24)
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" ...

# Pass 2: force-upgrade trl to the version we need
uv pip install --force-reinstall "trl==0.29.1"
```

### Suggested fix for upstream
Relax the upper bound to `trl>=0.15,<0.30` and test with current trl, or remove the
upper bound entirely and let the user pin if needed.

---

## 5. unsloth — Incompatibility with `transformers >= 5.4.0` (`auto_docstring`)

**Library**: `unsloth` (latest as of March 2026) × `transformers >= 5.4.0`
**File**: `unsloth/_utils.py`, `exec(config, globals())` pattern
**Affects**: Any user who installs `transformers>=5.4.0` with unsloth.

### Symptom
```
NameError: name 'auto_docstring' is not defined
```
During `FastLanguageModel.from_pretrained(...)`.

### Root cause
`transformers 5.4.0` introduced an `auto_docstring` decorator applied to model
`forward()` methods. When unsloth patches model code by running `exec(config, globals())`
on extracted source, `auto_docstring` is not in scope and the exec context crashes.

### Workaround
Pin `transformers==5.3.0`:
```bash
uv pip install --force-reinstall transformers==5.3.0
```

### Suggested fix for upstream
Either:
1. Inject `auto_docstring` into the exec globals, or
2. Strip `@auto_docstring` decorators from source before exec, or
3. Test the `unsloth_compiled_cache` approach against each new transformers minor release
   and update pinned bounds accordingly.

---

## 6. Vast.ai / NVIDIA CDI — `unresolvable CDI devices` on some host configurations

**Library / Platform**: Vast.ai container runtime (OCI)
**Affects**: Specific host machines (observed on Quebec/Canada region machines).

### Symptom
Container fails to start with:
```
OCI runtime create failed: error modifying OCI spec: failed to inject CDI devices:
unresolvable CDI devices D.eec56...
```

### Root cause
The host's NVIDIA container runtime has CDI (Container Device Interface) enabled but
improperly configured — the device UUID referenced in the CDI spec doesn't match the
physical device. This is a host-side misconfiguration on Vast.ai provider machines.

### Workaround
Select a different host machine or region. South Africa (ZA) machines with driver 580+
were not affected.

### Recommendation for users
When a Vast.ai instance fails to start immediately, SSH may never connect. Check the
`actual_status` field via the Vast.ai API: if the container stays in
`"loading" → "stopped"` loop after <60s, the CDI error is likely the cause. Destroy
and reprovision on a different host.

---

## 7. trl — `GRPOConfig.importance_sampling_level` added in 0.29.x only

**Library**: `trl`
**Affects**: Code using `importance_sampling_level` parameter of `GRPOConfig` with trl < 0.29.

### Symptom
```
TypeError: GRPOConfig.__init__() got an unexpected keyword argument 'importance_sampling_level'
```

### Context
`importance_sampling_level` was added to `GRPOConfig` in trl 0.29.x to control
GRPO's importance-sampling clip. This is a new feature (not a bug), but it's worth
noting for anyone targeting trl < 0.29 who reads GRPO tutorial code using it.

---

*Last updated: 2026-03-28*
*Context: healthcare-fraud GSPO training, Vast.ai RTX 3090, Ubuntu 22.04,*
*unsloth+trl==0.29.1+transformers==5.3.0+torch 2.11.0+cu128*
