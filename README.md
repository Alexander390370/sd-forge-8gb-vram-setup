# Running Stable Diffusion Forge on 8GB VRAM (RTX 4060)

> Deploying a fully local, uncensored image generation pipeline on an 8GB laptop GPU via WSL2, with a focus on model selection, VRAM budgeting, and the dependency hell that comes with it.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform: WSL2](https://img.shields.io/badge/Platform-WSL2-blue)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)

> **Status**: Field notes / engineering log. Not a maintained tool. The goal is to document what actually worked (and what didn't) so the next person saves a few hours.

## Overview

This repo documents running **Stable Diffusion WebUI Forge** on an RTX 4060 Laptop (8GB VRAM) inside WSL2. It covers:

- Why Forge over A1111 (VRAM management on low-end GPUs)
- Which community models are worth downloading (and where to get them without hitting dead links)
- The CUDA / PyTorch / xformers / triton version dance
- Patching xformers source for triton 3.x compatibility
- Choosing sampling parameters that don't blow VRAM

The goal is a working local pipeline that is **fully offline after setup** and **not subject to platform content filters** — the model files are community-finetuned weights, and the frontend runs with safe-check disabled.

> ⚠️ **Note on `--disable-safe-unpickle`**: this flag disables Forge's model load-time safety validation. Model files are Python pickles; a malicious one can execute arbitrary code when loaded. **Only use community models from sources you trust** (e.g. hf-mirror, well-known Civitai creators). The reason this flag is required here is that community-finetuned SDXL/SD1.5 models often don't pass the default safe-unpickle check.

## Hardware & Environment

- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM)
- **OS**: Windows 11 + WSL2 (Ubuntu 22.04)
- **Frontend**: [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)
- **Python**: 3.10 (via Conda)
- **Key runtime**: PyTorch 2.12 + CUDA 12.1, xformers (patched), triton 3.7.1

## Quick Start

```bash
# 1. Create the Conda environment
conda create -n sd-forge python=3.10 -y
conda activate sd-forge

# 2. (WSL2 only) Add the WSL CUDA library path — required for some pip wheels
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# 3. Clone Forge
cd ~
git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git
cd stable-diffusion-webui-forge

# 4. Launch with VRAM-friendly flags
python launch.py --api --listen --medvram --disable-safe-unpickle --port 7860
```

> If `git clone` times out against `github.com`, prepend a mirror: `git clone https://ghproxy.net/https://github.com/lllyasviel/stable-diffusion-webui-forge.git`.

**Expected result**: after launch, the terminal prints `Running on local URL: http://0.0.0.0:7860`. Open the WebUI, load a checkpoint from the model dropdown, and generate a 512×768 test image. If it completes in ~10-12s without OOM, you're in the sweet spot.

> The `--disable-safe-unpickle` flag disables Forge's model load-time safety check. See the warning in the Overview for what this implies.

Once the server starts, open `http://localhost:7860` (or the WSL IP for external clients — see the networking section).

## Model Selection

For 8GB VRAM, two paths are viable:

### Path A: SD 1.5 models (recommended for first run)

- **Realistic Vision V5.1** — the current go-to for photorealistic portraits.
- **DreamShaper 8** — stylized / painterly / versatile.
- **AnythingV5** — anime style, small footprint.

SD 1.5 models are ~4.2GB (fp16) and run comfortably at 512×768 on an 8GB card.

### Path B: SDXL models (tighter but workable)

- **Juggernaut XL** — the community's workhorse for photoreal SDXL.
- **Pony Diffusion V6 XL** — anime / versatile, massive LoRA ecosystem.
- **UnrealVision XL** — photoreal with strong detail, but pushes 8GB hard.

SDXL models are 6.5–7GB and require `--medvram` plus careful resolution choices.

### Download Sources

- **Hugging Face mirror** (`hf-mirror.com`) — fastest and most reliable. Example:
  ```bash
  cd ~/stable-diffusion-webui-forge/models/Stable-diffusion/
  aria2c -x 16 -s 16 -c "https://hf-mirror.com/scenario-labs/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors"
  ```
- **Civitai** — largest model catalog, but subject to network throttling and intermittent 403s from some regions.
- **ModelScope** — good for Qwen-family and Chinese-community uploads.

**Drop models into**: `~/stable-diffusion-webui-forge/models/Stable-diffusion/`

## Performance Tuning

### Startup flags

| Flag | What it does |
|------|--------------|
| `--medvram` | Offloads model components during generation. Slight speed cost, big VRAM savings. |
| `--lowvram` | Aggressive offload. Use only if `--medvram` still OOMs. |
| `--disable-safe-unpickle` | Skips the model load-time safety check. |
| `--api` | Exposes the REST API at `/docs`. |

### Sampling parameters (SD 1.5, 8GB)

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Sampler | `DPM++ 2M Karras` | Clean output, fast convergence. |
| Steps | `20-25` | Diminishing returns past 25. |
| CFG Scale | `6-7` | Above 8 often oversaturates on Realistic Vision. |
| Resolution | `512×768` | Portrait ratio; avoid 1024×1024 on 8GB. |
| Batch count / size | `1 / 1` | Batched generation on 8GB will OOM. |

### Hires. fix — use with care

Hires. fix at 2× multiplies effective resolution by 4. On 8GB, this is where most OOM crashes happen. The `Latent` upscaler is particularly VRAM-hungry — it re-runs the diffusion process at the target resolution, which is 4x the pixel count of a 2x upscale. **Avoid `Latent` on 8GB.**

- **Recommended**: turn Hires. fix **off** for the first pass. Generate 4 candidates at 512×768 (~10s each), pick the best, then send to `img2img` with denoising strength `0.3` at target resolution.
- If you insist on Hires. fix in one pass: use `R-ESRGAN 4x+` upscaler (not `Latent`), upscale by `1.5×` only, and set Hires steps to `10-12`.

## Gotchas & Fixes

### 1. `ModuleNotFoundError: No module named 'pkg_resources'`

**Cause**: `setuptools >= 82` removed `pkg_resources`. Forge's install path still depends on it during `openai/CLIP` build.

**Fix**:
```bash
pip install setuptools==69.5.1
pip install git+https://github.com/openai/CLIP.git --no-build-isolation
```
The `--no-build-isolation` flag is critical: it forces pip to use the downgraded setuptools instead of spinning up a fresh build env with the latest one.

### 2. `git clone` of CLIP times out

**Cause**: `github.com` is unreliable in some regions.

**Fix**: Set `CLIP_PACKAGE` to a mirror URL, or download the zip manually:
```bash
cd /tmp
wget https://ghproxy.net/https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip -O clip.zip
pip install /tmp/clip.zip --no-build-isolation
```

### 3. `ValueError: numpy.dtype size changed`

**Cause**: `scikit-image` was compiled against NumPy 1.x but you have NumPy 2.x.

**Fix**:
```bash
pip install "numpy<2"
```

### 4. `TypeError: JITCallable._set_src() takes 1 positional argument but 2 were given`

**Cause**: `xformers` calls `triton.runtime.jit.JITCallable._set_src()`, but the signature changed in triton 3.x.

**Fix**: Patch xformers source directly:
```bash
find ~ -name "vararg_kernel.py" -path "*xformers/triton*" \
  -exec sed -i 's/jitted_fn.src = new_src/jitted_fn._unsafe_update_src(new_src)/' {} \;
```

> **Note**: this patch targets triton 3.x with the specific xformers version bundled with Forge as of late 2026. It may break after future xformers / triton upgrades. If a newer Forge release already ships a compatible xformers, skip this step entirely.

### 5. `ModuleNotFoundError: No module named 'joblib'`

**Cause**: One of Forge's built-in extensions (`soft-inpainting`) imports `joblib` but doesn't declare it as a dependency.

**Fix**:
```bash
pip install joblib
```

### 6. Windows client can't reach `http://127.0.0.1:7860`

**Cause**: WSL2 uses NAT by default — Windows `localhost` is not the same as WSL `localhost`.

**Fix**: Use `ip addr` inside WSL to find the LAN IP (e.g. `192.168.x.x`), then use `http://<WSL_IP>:7860` from Windows clients. Alternatively, enable WSL2 mirrored networking in `%USERPROFILE%\.wslconfig`:
```ini
[wsl2]
networkingMode=mirrored
```

### 7. OOM even with `--medvram`

**Cause**: Another GPU-heavy process (LM Studio, Ollama, another Forge instance) is holding VRAM.

**Fix**: Close everything else. On 8GB, there is no room for two GPU workloads.

## Benchmarks

**Conditions**: Idle GPU, `--medvram`, `Realistic_Vision_V5.1`, 512×768, 20 steps, DPM++ 2M Karras.

| Metric | Value |
|--------|-------|
| Single image (no hires fix) | ~10-12s |
| Single image (1.5× hires, R-ESRGAN) | ~25-35s |
| SDXL (Juggernaut XL, 768×768) | ~45-60s |
| Peak VRAM usage (SD 1.5, no hires) | ~4.5GB |
| Peak VRAM usage (SDXL, 768×768) | ~7.3GB |

## Known Limitations

- **WSL2 GPU memory is shared with the Windows host.** The Windows desktop compositor and any GPU-accelerated browser tabs will eat into your available VRAM. If your numbers don't match the benchmarks above, check what else is running on Windows first.
- **SDXL on 8GB is tight.** Even at 768×768 with `--medvram`, a background Chrome tab with GPU acceleration can push you over the edge.
- **No batching.** Batch size > 1 will OOM on 8GB for anything above 512×512.
- **Hires. fix is the crash point.** Most reported OOMs on this hardware happen during the upscale pass, not the base generation.
- **Civitai downloads may fail intermittently.** Use the hf-mirror path when possible; Civitai's CDN has region-dependent throttling.
- **No auth on the Forge web UI.** If you expose it on the LAN, anyone on the network can generate images on your GPU. Fine for home use, not for shared environments.

## Related Work

The same 8GB RTX 4060, running different workloads:

- **[Bonsai-27B-8GB-VRAM-Setup](https://github.com/Alexander390370/Bonsai-27B-8GB-VRAM-Setup)** — 27B quantized LLM at 64K context. Shares the same VRAM budgeting mindset.
- **[OmniForge-Data-Annotation](https://github.com/Alexander390370/OmniForge-Data-Annotation)** — full-modal data annotation. Generated images can be used for dataset augmentation.
- **[esp32-edge-ai-security](https://github.com/Alexander390370/esp32-edge-ai-security)** — edge AI on the hardware side of the same pipeline.
- **[esp32-pwm-fan-controller](https://github.com/Alexander390370/esp32-pwm-fan-controller)** — hardware-side firmware project.

## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.
