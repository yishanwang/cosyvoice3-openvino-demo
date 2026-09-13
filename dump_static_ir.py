#!/usr/bin/env python3
"""Capture the ACTUAL input shapes each CosyVoice3 OV IR component sees
during a real CPU inference run, then dump a statically-reshaped copy of
each IR (using those observed shapes) to a separate output folder --
these are the shapes an NPU deployment of these IRs would need to be
compiled with (NPU requires static shapes; dynamic-shape IR as shipped in
the model package can't be compiled on NPU directly).

Does NOT modify the original OV IR package. Purely observational: wraps
each compiled model with a thin shape-recording proxy, runs one real
zero-shot synthesis end-to-end on CPU (so all shapes reflect genuine
runtime values, not guesses), then for every distinct shape seen per
component, reads a fresh copy of that component's IR, reshapes it to that
exact shape, and saves it to the output directory.

Usage:
    python dump_static_ir.py \\
        --cosyvoice-src /path/to/CosyVoice \\
        --model-dir /path/to/cosyvoice3-0.5b-2512 \\
        --output-dir /path/to/output \\
        --tts-text "..." --prompt-text "..." --prompt-wav ref.wav
"""
import argparse
import json
import os
import sys

import numpy as np


def shape_of(x):
    return tuple(np.asarray(x).shape)


class ShapeRecorder:
    """Wraps a callable OV compiled model (list- or dict-input style),
    recording every distinct input-shape combination it's called with,
    then forwarding the call unchanged to the real model."""

    def __init__(self, real_model, name):
        self.real_model = real_model
        self.name = name
        self.seen = []  # list of shape-records (dict name->shape, or tuple of shapes)

    def _record(self, inputs):
        if isinstance(inputs, dict):
            rec = tuple(sorted((k, shape_of(v)) for k, v in inputs.items()))
        elif isinstance(inputs, (list, tuple)):
            rec = tuple(shape_of(v) for v in inputs)
        else:
            rec = (shape_of(inputs),)
        if rec not in self.seen:
            self.seen.append(rec)

    def __call__(self, inputs):
        self._record(inputs)
        return self.real_model(inputs)


class InferRequestShapeRecorder:
    """Wraps an ov.InferRequest's .infer(inputs) (dict-based, stateful LLM
    call convention) the same way, recording shapes then forwarding."""

    def __init__(self, real_request, name):
        self.real_request = real_request
        self.name = name
        self.seen = []

    def infer(self, inputs):
        rec = tuple(sorted((k, shape_of(v)) for k, v in inputs.items()))
        if rec not in self.seen:
            self.seen.append(rec)
        return self.real_request.infer(inputs)

    def __getattr__(self, item):
        # Delegate everything else (get_tensor, reset_state, ...) to the real request.
        return getattr(self.real_request, item)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cosyvoice-src", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tts-text", required=True)
    p.add_argument("--prompt-text", required=True)
    p.add_argument("--prompt-wav", required=True)
    args = p.parse_args()

    sys.path.insert(0, args.cosyvoice_src)
    sys.path.insert(0, os.path.join(args.cosyvoice_src, "third_party/Matcha-TTS"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
    os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

    import openvino as ov
    import ov_cosyvoice_helper as helper

    core = ov.Core()

    print("⌛ Loading baseline pipeline on CPU...")
    ov_cosyvoice = helper.OVCosyVoice3(model_dir=args.model_dir, device="CPU")
    print("✅ Baseline pipeline loaded on CPU")

    # (component name, xml relpath under model_dir, the live object+attribute to wrap)
    components = {
        "text_embeddings": (helper.TEXT_EMBEDDINGS_PATH, ov_cosyvoice.model.llm, "text_embeddings"),
        "speech_embeddings": (helper.SPEECH_EMBEDDINGS_PATH, ov_cosyvoice.model.llm, "speech_embeddings"),
        "flow_embeddings": (helper.FLOW_EMBEDDINGS_PATH, ov_cosyvoice.model.flow, "flow_embeddings"),
        "flow_estimator": (helper.FLOW_ESTIMATOR_PATH, ov_cosyvoice.model.flow, "flow_estimator"),
        "hift": (helper.HIFT_PATH, ov_cosyvoice.model.hift, "hift"),
        "campplus": ("campplus.onnx", ov_cosyvoice.frontend, "campplus_model"),
        "speech_tokenizer": ("speech_tokenizer_v3.onnx", ov_cosyvoice.frontend, "speech_tokenizer_model"),
    }

    recorders = {}
    for name, (_, obj, attr) in components.items():
        real = getattr(obj, attr)
        rec = ShapeRecorder(real, name)
        setattr(obj, attr, rec)
        recorders[name] = rec

    # LLM: stateful, called via a pre-created InferRequest, not a plain CompiledModel.__call__.
    llm_wrapper = ov_cosyvoice.model.llm
    llm_recorder = InferRequestShapeRecorder(llm_wrapper.llm_request, "llm")
    llm_wrapper.llm_request = llm_recorder
    recorders["llm"] = llm_recorder

    print(f"⌛ Running real zero-shot synthesis to capture actual shapes: {args.tts_text!r}")
    chunks = []
    for output in ov_cosyvoice.inference_zero_shot(args.tts_text, args.prompt_text, args.prompt_wav, stream=False):
        chunks.append(output["tts_speech"])
    print("✅ Synthesis complete\n")

    print("Distinct input shapes observed per component:")
    for name, rec in recorders.items():
        print(f"  {name}: {len(rec.seen)} distinct shape(s)")
        for s in rec.seen:
            print(f"    {s}")

    # --- Dump static IR per distinct observed shape ---
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {}

    def reshape_spec_from_record(rec, input_names_in_order=None):
        """Turn a recorded shape-tuple into a reshape() argument."""
        if isinstance(rec[0], tuple) and len(rec[0]) == 2 and isinstance(rec[0][0], str):
            # dict-style record: tuple of (name, shape) pairs
            return {k: list(v) for k, v in rec}
        # positional list-style record: one shape per input, in order
        if input_names_in_order is None:
            return [list(s) for s in rec]
        return {n: list(s) for n, s in zip(input_names_in_order, rec)}

    for name, (xml_relpath, _, _) in components.items():
        rec = recorders[name]
        xml_path = os.path.join(ov_cosyvoice.ov_model_dir, xml_relpath)
        model_for_names = core.read_model(xml_path)
        input_names = [i.get_any_name() for i in model_for_names.inputs]

        manifest[name] = {"source_xml": xml_relpath, "static_variants": []}
        for idx, rec_entry in enumerate(rec.seen):
            reshape_spec = reshape_spec_from_record(rec_entry, input_names)
            m = core.read_model(xml_path)
            m.reshape(reshape_spec)
            suffix = f"_shape{idx}" if len(rec.seen) > 1 else ""
            out_name = f"{name}{suffix}.xml"
            out_path = os.path.join(args.output_dir, out_name)
            # compress_to_fp16=False: preserve whatever precision the source
            # model already has -- do NOT let save_model's default FP16
            # compression silently downcast full-precision sources (this bit
            # speech_tokenizer_v3 before: FP16 flips ~80% of its discrete
            # token-index outputs, see cosyvoice3-openvino-conversion's README).
            ov.save_model(m, out_path, compress_to_fp16=False)
            manifest[name]["static_variants"].append({"reshape": reshape_spec, "file": out_name})
            print(f"✅ Saved {out_path}  (reshape={reshape_spec})")

    # LLM: autoregressive -- attention_mask grows by exactly 1 every decode
    # step, so a real run produces one near-identical shape PER GENERATED
    # TOKEN (e.g. 129 shapes for a 128-token generation). Dumping a full
    # ~700MB static IR per step is impractical and not how a real static
    # NPU deployment would work anyway (that needs one FIXED max-context
    # shape with attention-mask-based padding, decided ahead of time).
    # Instead, dump just the two structurally distinct shape categories:
    #   - "prefill": the one call with inputs_embeds seq_len > 1 (encodes
    #     the whole initial prompt in one shot)
    #   - "decode_max_context": a decode step (inputs_embeds seq_len == 1)
    #     at the LARGEST attention_mask length observed -- the worst-case
    #     single-new-token shape for this run.
    llm_xml_path = os.path.join(ov_cosyvoice.ov_model_dir, helper.LANGUAGE_PATH)
    manifest["llm"] = {
        "source_xml": helper.LANGUAGE_PATH,
        "note": (
            f"{len(llm_recorder.seen)} distinct shapes were observed across this run "
            "(one per generated token, since attention_mask grows by 1 each decode "
            "step) -- only the prefill shape and the max-context decode shape are "
            "dumped; see dump_static_ir.py comments for why."
        ),
        "all_observed_shapes": [{k: list(v) for k, v in rec} for rec in llm_recorder.seen],
        "static_variants": [],
    }

    def embeds_seqlen(rec):
        return dict(rec)["inputs_embeds"][1]

    prefill_recs = [rec for rec in llm_recorder.seen if embeds_seqlen(rec) > 1]
    decode_recs = [rec for rec in llm_recorder.seen if embeds_seqlen(rec) == 1]
    to_dump = []
    if prefill_recs:
        to_dump.append(("llm_prefill.xml", prefill_recs[0]))
    if decode_recs:
        max_ctx_rec = max(decode_recs, key=lambda rec: dict(rec)["attention_mask"][1])
        to_dump.append(("llm_decode_max_context.xml", max_ctx_rec))

    for out_name, rec_entry in to_dump:
        reshape_spec = {k: list(v) for k, v in rec_entry}
        m = core.read_model(llm_xml_path)
        m.reshape(reshape_spec)
        out_path = os.path.join(args.output_dir, out_name)
        ov.save_model(m, out_path, compress_to_fp16=False)
        manifest["llm"]["static_variants"].append({"reshape": reshape_spec, "file": out_name})
        print(f"✅ Saved {out_path}  (reshape={reshape_spec})")

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Wrote manifest.json to {args.output_dir}")


if __name__ == "__main__":
    main()
