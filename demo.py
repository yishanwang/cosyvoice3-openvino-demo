#!/usr/bin/env python3
"""CosyVoice3 OpenVINO IR demo -- zero-shot TTS with per-component device
selection (CPU / NPU / GPU / AUTO).

NPU only supports static input shapes, so any component targeted at NPU is
reshaped to a static shape before compiling (see device_loader.py for which
components support transparent padding vs. require exact-shape input, and
why). Any component that fails to compile/reshape on its requested device
automatically falls back to CPU (with a warning) -- this is deliberate: not
every IR is expected to run on NPU today.

Requires:
  - A checkout of https://github.com/FunAudioLLM/CosyVoice (for the
    `cosyvoice` / `matcha` Python packages used by the frontend + inference
    orchestration -- NOT for the model weights, which come entirely from
    the OV IR package below).
  - The OV IR model package, e.g. cloned from
    https://github.com/yishanwang/npu_prc_ovir_model_package
    (`cosyvoice3-0.5b-2512/` subfolder).

Example (all-CPU, safe baseline):
    python demo.py \\
        --cosyvoice-src /path/to/CosyVoice \\
        --model-dir /path/to/npu_prc_ovir_model_package/cosyvoice3-0.5b-2512 \\
        --tts-text "Hello there, this is a test." \\
        --prompt-text "This is the reference speaker's transcript." \\
        --prompt-wav /path/to/reference.wav \\
        --output out.wav

Example (try a few components on NPU, everything else CPU):
    python demo.py \\
        --cosyvoice-src /path/to/CosyVoice \\
        --model-dir /path/to/cosyvoice3-0.5b-2512 \\
        --tts-text "..." --prompt-text "..." --prompt-wav ref.wav --output out.wav \\
        --text-embeddings-device NPU --speech-embeddings-device NPU \\
        --hift-device NPU --hift-static-mel-len 512
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--cosyvoice-src", required=True, help="Path to a checkout of github.com/FunAudioLLM/CosyVoice")
    p.add_argument("--model-dir", required=True, help="Path to the cosyvoice3-0.5b-2512 OV IR package directory")

    p.add_argument("--tts-text", help="Text to synthesize")
    p.add_argument("--prompt-text", help="Transcript of the reference/prompt audio (zero-shot mode)")
    p.add_argument("--prompt-wav", help="Path to the reference/prompt wav file (zero-shot mode)")
    p.add_argument("--output", default="out.wav", help="Output wav path (default: out.wav)")

    p.add_argument("--list-devices", action="store_true", help="Print available OpenVINO devices and exit")

    # Per-IR-file device selection -- this is the core ask: every OV IR
    # file used by the pipeline gets its own independent device choice.
    for name in [
        "llm", "text-embeddings", "speech-embeddings", "flow-embeddings",
        "flow-estimator", "hift", "campplus", "speech-tokenizer",
    ]:
        p.add_argument(f"--{name}-device", default="CPU", choices=["CPU", "NPU", "GPU", "AUTO"],
                        help=f"OpenVINO device for the {name.replace('-', '_')} IR (default: CPU)")

    p.add_argument("--fallback-device", default="CPU", help="Device to fall back to if a component fails on its requested device (default: CPU)")
    p.add_argument("--npu-config", default="{}", help="Extra OpenVINO NPU compile config, as a JSON object string (default: '{}')")

    # Static shapes -- only used for components whose --*-device is NPU.
    p.add_argument("--text-embeddings-static-len", type=int, default=128, help="Static text-token sequence length for NPU (default: 128)")
    p.add_argument("--speech-embeddings-static-len", type=int, default=512, help="Static speech-token sequence length for NPU (default: 512)")
    p.add_argument("--campplus-static-len", type=int, default=400, help="Static mel-frame length for campplus on NPU (default: 400, must be EXACT at inference time -- see README)")
    p.add_argument("--speech-tokenizer-static-len", type=int, default=400, help="Static mel-frame length for speech_tokenizer on NPU (default: 400, must be EXACT at inference time -- see README)")
    p.add_argument("--flow-embeddings-static-token-len", type=int, default=128, help="Static token length for flow_embeddings on NPU (must be EXACT at inference time)")
    p.add_argument("--flow-embeddings-static-prompt-len", type=int, default=128, help="Static prompt_token length for flow_embeddings on NPU (must be EXACT at inference time)")
    p.add_argument("--flow-embeddings-static-spk-dim", type=int, default=192, help="Speaker-embedding dim for flow_embeddings static shape (default: 192, matches campplus output)")
    p.add_argument("--flow-estimator-static-mel-len", type=int, default=512, help="Static mel length for flow_estimator on NPU (must be EXACT at inference time)")
    p.add_argument("--hift-static-mel-len", type=int, default=512, help="Static mel length for hift on NPU (uses OVHiFT's own built-in pad/trim -- safe, unlike the other EXACT-shape components)")

    return p.parse_args()


def main():
    args = parse_args()

    sys.path.insert(0, args.cosyvoice_src)
    sys.path.insert(0, os.path.join(args.cosyvoice_src, "third_party/Matcha-TTS"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
    os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

    import openvino as ov
    import ov_cosyvoice_helper as helper
    from device_loader import load_component

    core = ov.Core()

    if args.list_devices:
        print("Available OpenVINO devices:", core.available_devices)
        return

    if not (args.tts_text and args.prompt_text and args.prompt_wav):
        raise SystemExit("--tts-text, --prompt-text and --prompt-wav are all required (unless --list-devices)")

    npu_config = json.loads(args.npu_config)

    # 1. Build the full pipeline entirely on CPU first -- a safe, always-
    #    working baseline. Every component is then independently
    #    recompiled below onto its user-requested device.
    print("⌛ Loading baseline pipeline on CPU...")
    ov_cosyvoice = helper.OVCosyVoice3(model_dir=args.model_dir, device="CPU")
    print("✅ Baseline pipeline loaded on CPU")

    resolved_devices = {}

    def recompile(component, xml_relpath, requested_device, static_shape):
        if requested_device.upper() == "CPU":
            resolved_devices[component] = "CPU"
            return None  # keep the CPU baseline compiled model already loaded
        xml_path = os.path.join(ov_cosyvoice.ov_model_dir, xml_relpath)
        model, actual_device = load_component(
            core, xml_path, component, device=requested_device,
            static_shape=static_shape, npu_config=npu_config,
            fallback_device=args.fallback_device,
        )
        resolved_devices[component] = actual_device
        return model

    # --- llm / text_embeddings / speech_embeddings ---
    llm_model = recompile("llm", helper.LANGUAGE_PATH, args.llm_device, None)  # no static-shape support (stateful, dynamic KV-cache)
    if llm_model is not None:
        ov_cosyvoice.model.llm.llm = llm_model

    text_emb_model = recompile(
        "text_embeddings", helper.TEXT_EMBEDDINGS_PATH, args.text_embeddings_device,
        [1, args.text_embeddings_static_len],
    )
    if text_emb_model is not None:
        ov_cosyvoice.model.llm.text_embeddings = text_emb_model

    speech_emb_model = recompile(
        "speech_embeddings", helper.SPEECH_EMBEDDINGS_PATH, args.speech_embeddings_device,
        [1, args.speech_embeddings_static_len],
    )
    if speech_emb_model is not None:
        ov_cosyvoice.model.llm.speech_embeddings = speech_emb_model

    # --- flow_embeddings / flow_estimator ---
    flow_emb_shape = {
        "token": [1, args.flow_embeddings_static_token_len],
        "token_len": [1],
        "prompt_token": [1, args.flow_embeddings_static_prompt_len],
        "prompt_token_len": [1],
        "embedding": [1, args.flow_embeddings_static_spk_dim],
    }
    flow_emb_model = recompile(
        "flow_embeddings", helper.FLOW_EMBEDDINGS_PATH, args.flow_embeddings_device, flow_emb_shape
    )
    if flow_emb_model is not None:
        ov_cosyvoice.model.flow.flow_embeddings = flow_emb_model

    mel_len = args.flow_estimator_static_mel_len
    flow_est_shape = {
        "x": [2, 80, mel_len], "mask": [2, 1, mel_len], "mu": [2, 80, mel_len],
        "t": [2], "spks": [2, 80], "cond": [2, 80, mel_len],
    }
    flow_est_model = recompile(
        "flow_estimator", helper.FLOW_ESTIMATOR_PATH, args.flow_estimator_device, flow_est_shape
    )
    if flow_est_model is not None:
        ov_cosyvoice.model.flow.flow_estimator = flow_est_model

    # --- hift: reuse OVHiFT's own official static-shape handling ---
    if args.hift_device.upper() == "CPU":
        resolved_devices["hift"] = "CPU"
    else:
        hift_xml = os.path.join(ov_cosyvoice.ov_model_dir, helper.HIFT_PATH)
        try:
            new_hift = helper.OVHiFT(
                model_path=hift_xml, device=args.hift_device.upper(),
                hift_input_len=args.hift_static_mel_len if args.hift_device.upper() == "NPU" else 0,
            )
            ov_cosyvoice.model.hift = new_hift
            resolved_devices["hift"] = args.hift_device.upper()
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[hift] failed on {args.hift_device} ({e!r}) -- falling back to {args.fallback_device}.")
            resolved_devices["hift"] = args.fallback_device

    # --- campplus / speech_tokenizer (frontend) ---
    campplus_model = recompile(
        "campplus", "campplus.onnx", args.campplus_device, [1, args.campplus_static_len, 80]
    )
    if campplus_model is not None:
        ov_cosyvoice.frontend.campplus_model = campplus_model

    speech_tok_shape = [
        [1, 128, args.speech_tokenizer_static_len],
        [1],
    ]
    if args.speech_tokenizer_device.upper() == "CPU":
        resolved_devices["speech_tokenizer"] = "CPU"
    else:
        xml_path = os.path.join(ov_cosyvoice.ov_model_dir, "speech_tokenizer_v3.onnx")
        # speech_tokenizer has 2 inputs (feats, feats_length) -- reshape needs a dict keyed by input name.
        model_r = core.read_model(xml_path)
        input_names = [i.get_any_name() for i in model_r.inputs]
        shape_dict = {input_names[0]: speech_tok_shape[0], input_names[1]: speech_tok_shape[1]}
        tok_model, actual = load_component(
            core, xml_path, "speech_tokenizer", device=args.speech_tokenizer_device,
            static_shape=shape_dict, npu_config=npu_config, fallback_device=args.fallback_device,
        )
        resolved_devices["speech_tokenizer"] = actual
        ov_cosyvoice.frontend.speech_tokenizer_model = tok_model

    print("\nResolved device per IR file (after any automatic fallback):")
    for k, v in resolved_devices.items():
        print(f"  {k:20s} -> {v}")
    print()

    # 2. Run zero-shot TTS inference.
    print(f"⌛ Synthesizing: {args.tts_text!r}")
    chunks = []
    for output in ov_cosyvoice.inference_zero_shot(args.tts_text, args.prompt_text, args.prompt_wav, stream=False):
        chunks.append(output["tts_speech"].numpy() if hasattr(output["tts_speech"], "numpy") else np.asarray(output["tts_speech"]))
    audio = np.concatenate(chunks, axis=-1).squeeze()

    import soundfile as sf
    sf.write(args.output, audio, ov_cosyvoice.sample_rate)
    print(f"✅ Wrote {args.output} ({audio.shape[-1] / ov_cosyvoice.sample_rate:.2f}s @ {ov_cosyvoice.sample_rate} Hz)")


if __name__ == "__main__":
    main()
