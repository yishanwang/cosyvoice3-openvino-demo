# CosyVoice3 OpenVINO Demo

A Python demo for [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
zero-shot TTS running entirely on OpenVINO IR, with **independent CPU/NPU/GPU
device selection for every OV IR file** in the pipeline.

Uses the OV IR package from
[yishanwang/npu_prc_ovir_model_package](https://github.com/yishanwang/npu_prc_ovir_model_package)'s
`cosyvoice3-0.5b-2512/` folder, produced by
[yishanwang/cosyvoice3-openvino-conversion](https://github.com/yishanwang/cosyvoice3-openvino-conversion).

**Validated end-to-end**: ran a real zero-shot synthesis
(`FunAudioLLM/CosyVoice`'s own `asset/zero_shot_prompt.wav` sample) both
all-CPU and with `NPU` requested for 5 of the 8 components (which — since
this dev machine's NPU compiler backend isn't installed — correctly
triggered automatic fallback to CPU for each, with clear warnings) —
both runs produced valid, non-silent 24kHz audio. See "Known limitations"
below for exactly what was and wasn't validated.

## Why per-IR-file device selection?

NPU support across a multi-component pipeline like this rolls out
unevenly — some IR files convert and run cleanly on NPU, others don't (yet).
Rather than an all-or-nothing `--device` flag, every one of the 8 OV IR
files gets its own flag, so you can move components to NPU one at a time as
support matures, while everything else stays on a known-good device:

| Flag | IR file |
|:-----|:--------|
| `--llm-device` | `openvino_model.xml` (autoregressive LLM, stateful KV-cache) |
| `--text-embeddings-device` | `openvino_text_embeddings_model.xml` |
| `--speech-embeddings-device` | `openvino_speech_embeddings_model.xml` |
| `--flow-embeddings-device` | `openvino_flow_embeddings_model.xml` |
| `--flow-estimator-device` | `openvino_flow_estimator_model.xml` |
| `--hift-device` | `openvino_hift_model.xml` |
| `--campplus-device` | `openvino_campplus_model.xml` |
| `--speech-tokenizer-device` | `openvino_speech_tokenizer_v3_model.xml` |

**Any component that fails to compile/reshape on its requested device
automatically falls back to `--fallback-device` (default `CPU`), with a
warning** — this is deliberate, not a bug: it's exactly the "maybe some IRs
aren't NPU-ready yet" scenario this demo is built to handle gracefully.

## NPU static-shape requirement

NPU only supports static input shapes. `demo.py` reshapes any
NPU-targeted component to a static shape before compiling. Two different
strategies are used, depending on what was **empirically verified** on
this model (see "NPU static-shape notes" below for the actual experiments
— important: **this development environment had no physical NPU**, so
"NPU" behavior below was verified by reshaping to a static shape and
running on CPU, which enforces the same static-shape numerical contract
NPU requires, without requiring NPU silicon):

1. **Pad-safe** (`text_embeddings`, `speech_embeddings`): the sequence is
   transparently zero-padded to the static length and the output is
   sliced back to the real length. Verified **bit-exact** vs. the dynamic
   (unpadded) computation — these are pure per-position embedding lookups
   with no cross-position mixing.
2. **Exact-shape-only** (`campplus`, `speech_tokenizer`, `flow_embeddings`,
   `flow_estimator`): reshaping to a static shape is supported, but you
   must supply data at **exactly** that static shape — no padding is
   applied. Padding was tested and found **unsafe** for these (see below).
3. **`hift`**: reuses `OVHiFT`'s own official static-shape handling
   (`hift_input_len`, from the upstream `openvino_notebooks` helper) —
   already pads/trims mel frames correctly, not reimplemented here.
4. **`llm`**: autoregressive with a dynamic-length stateful KV-cache;
   static reshaping is genuinely out of scope for this demo. Device
   selection is exposed, but whether NPU actually works depends on the
   OpenVINO NPU plugin's own support for this pattern — **not validated**
   in this repo. Automatic CPU fallback still applies.

### NPU static-shape notes (the actual experiments)

Run in this repo's dev environment (CPU-only hardware — no NPU available):

- **`text_embeddings` / `speech_embeddings`**: reshaped to a static
  `[1, 64]` / `[1, 128]` input, zero-padded a shorter real sequence into
  it, and compared the un-padded prefix of the output against a plain
  dynamic-shape run on the same input. **Bit-exact match.** Expected: pure
  `nn.Embedding`-style Gather, computed independently per position.
- **`campplus`**: reshaped to a static `[1, 400, 80]` input, zero-padded a
  250-frame real input into it, and compared against a dynamic-shape run
  on the same 250 frames. **Max abs diff ~1.24** (on outputs typically in
  `[-1.9, 1.9]`) — the statistics-pooling layer mixes information across
  the whole time axis, so padding measurably corrupts the result. Do not
  pad this component; use the exact expected length.
- **`speech_tokenizer_v3`**: attempted the same pad-and-compare test with
  a static `[1, 128, 400]` shape and a 300-frame real input (with
  `feats_length=300` telling the model the true length) — OpenVINO raised
  an internal shape-inference error (`Eltwise shape infer ... mismatch`)
  inside the model's self-attention blocks. The `feats_length` input is
  **not** a sufficient masking mechanism for arbitrary padding in this
  exported graph. Do not pad this component; use the exact expected length.
- **`flow_embeddings` / `flow_estimator`**: not empirically tested (no
  standalone PyTorch reference was loaded in this repo to compare
  against), but both are DiT/attention-style networks with the same
  cross-position-mixing pattern as `speech_tokenizer_v3` — treated the
  same way (exact-shape-only) out of caution.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

git clone https://github.com/yishanwang/npu_prc_ovir_model_package.git
# (needs git-lfs: git lfs install && git lfs pull, if not done automatically)
```

## Usage

All-CPU baseline (always works, good first check):

```bash
python demo.py \
  --cosyvoice-src ./CosyVoice \
  --model-dir ./npu_prc_ovir_model_package/cosyvoice3-0.5b-2512 \
  --tts-text "Hello there, this is a test of CosyVoice3 on OpenVINO." \
  --prompt-text "This is the reference speaker's transcript." \
  --prompt-wav ./reference.wav \
  --output out.wav
```

Try specific components on NPU (falls back to CPU automatically for any
that fail):

```bash
python demo.py \
  --cosyvoice-src ./CosyVoice \
  --model-dir ./npu_prc_ovir_model_package/cosyvoice3-0.5b-2512 \
  --tts-text "..." --prompt-text "..." --prompt-wav ./reference.wav --output out.wav \
  --text-embeddings-device NPU \
  --speech-embeddings-device NPU \
  --hift-device NPU --hift-static-mel-len 512
```

List available OpenVINO devices on your machine:

```bash
python demo.py --cosyvoice-src ./CosyVoice --model-dir ./npu_prc_ovir_model_package/cosyvoice3-0.5b-2512 --list-devices
```

`demo.py` prints a final "resolved device per IR file" summary after
loading, showing what actually ran where (post-fallback) — useful for
confirming which components genuinely ran on NPU vs. silently fell back.

See `python demo.py --help` for the full list of static-shape flags
(`--text-embeddings-static-len`, `--campplus-static-len`,
`--flow-estimator-static-mel-len`, etc.) — all default to reasonable
values but must be sized to your actual expected input lengths for the
exact-shape-only components.

## License

`demo.py` and `device_loader.py` are original work. `third_party/ov_cosyvoice_helper.py`
is vendored, unmodified, from
[`openvinotoolkit/openvino_notebooks`](https://github.com/openvinotoolkit/openvino_notebooks)
PR #3243 (Apache License 2.0) — see the file header for provenance.

## Known limitations / troubleshooting

- **Testing environment had no physical NPU compiler available** (the NPU
  plugin is registered, but `libopenvino_intel_npu_compiler_loader.so` is
  missing, so any real NPU compile raises
  `VCL compiler loading failed, aborting`). This turned out to be an ideal
  way to validate the automatic-fallback path for real: `demo.py` was run
  requesting `NPU` for `text_embeddings`, `speech_embeddings`, `campplus`,
  `speech_tokenizer`, and `hift` simultaneously — all five hit the missing
  NPU compiler at compile time, all five printed a clear warning and fell
  back to CPU, and the full zero-shot TTS pipeline still completed and
  produced valid, non-silent audio. The static-shape reshape math itself
  (separately, on CPU) was validated as described above. What is **not**
  validated is whether real NPU hardware/compiler accepts these reshaped
  models and produces correct output — if a component fails to compile on
  real NPU, it will automatically fall back to `--fallback-device` and
  print a warning; please report back what happened.
- **`OVCosyVoiceFrontEnd.__init__` needs network access to ModelScope**
  (unconditionally, regardless of `--*-device` flags): it downloads a
  Chinese/English text-normalization model (`pengzhendong/wetext`) via
  `modelscope`'s `snapshot_download()` on first run. In network-restricted
  environments this occasionally hit transient errors (`403 Authentication
  token does not exist`, or an SSL cert verification failure) that
  resolved themselves on a plain retry — if you see either, just re-run.
  Workarounds if it persists: ensure `HTTP_PROXY`/`HTTPS_PROXY` are
  exported if you're behind a corporate proxy, authenticate with
  ModelScope (`modelscope login`), or install the alternative `ttsfrd`
  package (checked first, before falling back to `wetext`) per
  CosyVoice's own setup instructions. Once downloaded, it's cached under
  `~/.cache/modelscope/` and this step is skipped on subsequent runs.


