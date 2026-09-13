"""Per-IR-file OpenVINO device selection + static-shape support for the
CosyVoice3 OV IR components, with automatic fallback if a component can't
compile/run on the requested device.

NPU only supports static input shapes. Two flavors of static-shape support
are offered here, based on what was empirically verified on this model's
components (see README.md "NPU static-shape notes" for the full writeup and
the exact experiments):

1. PAD-SAFE components (`text_embeddings`, `speech_embeddings`): verified
   bit-exact when the sequence axis is zero-padded to a fixed static length
   and the output is sliced back to the real length. These are pure
   per-position embedding lookups (`nn.Embedding`-style Gather) with no
   cross-position mixing, so padding the tail cannot affect earlier
   positions. Handled transparently by `PaddedCompiledModel` below.

2. EXACT-SHAPE components (`campplus`, `speech_tokenizer`, `flow_embeddings`,
   `flow_estimator`): reshaping to a static shape is supported, but
   zero-padding is NOT safe -- verified experimentally:
     - `campplus` (TDNN + statistics pooling speaker embedding): padding the
       time axis changed the pooled output by up to ~1.24 absolute (pooling
       mixes across the whole sequence).
     - `speech_tokenizer_v3` (transformer-based, self-attention blocks):
       padding raised an internal OpenVINO shape-inference error outright.
   `flow_embeddings`/`flow_estimator` were not empirically tested (no
   reference to compare against without the full flow.pt PyTorch module
   loaded) but share the same architectural pattern (DiT / attention-style
   mixing across the sequence axis), so are conservatively treated the same
   way: static reshape is supported, but callers must supply data at
   *exactly* the configured static shape -- no padding trick is applied.

`hift` already has its own official (openvino_notebooks-authored) static
shape handling via `hift_input_len` in `OVHiFT` -- reused as-is via
`demo.py`, not reimplemented here.

`llm` is autoregressive with a dynamic-length stateful KV-cache -- static
reshaping is out of scope for this demo. Device selection is still exposed
(`--llm-device`), but whether it actually runs on NPU today depends on the
OpenVINO NPU plugin's own support for this stateful pattern -- not
validated here (no NPU hardware was available in the environment this demo
was built in). Automatic fallback to CPU applies here too.
"""
import warnings

import numpy as np
import openvino as ov

# Axis (in the compiled model's single positional input) that is safe to
# zero-pad-and-trim without changing the result for the un-padded prefix.
PAD_SAFE_AXIS = {
    "text_embeddings": 1,
    "speech_embeddings": 1,
}


class PaddedCompiledModel:
    """Wraps a statically-reshaped CompiledModel for a PAD-SAFE component.

    Transparently pads the single positional input along `axis` up to
    `static_len`, runs inference, and slices the first output back down to
    the real (pre-padding) length along the same axis. Only use this for
    components verified pad-safe (see module docstring) -- do NOT use it
    for components with cross-position mixing (attention, pooling, etc.).
    """

    def __init__(self, compiled_model: ov.CompiledModel, static_len: int, axis: int = 1):
        self.compiled_model = compiled_model
        self.static_len = static_len
        self.axis = axis

    def __call__(self, inputs):
        arr = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        arr = np.asarray(arr)
        real_len = arr.shape[self.axis]
        if real_len > self.static_len:
            raise ValueError(
                f"Input length {real_len} exceeds the configured static NPU "
                f"shape {self.static_len} along axis {self.axis}. Increase "
                f"the --*-static-len value for this component."
            )
        pad_width = [(0, 0)] * arr.ndim
        pad_width[self.axis] = (0, self.static_len - real_len)
        padded = np.pad(arr, pad_width)
        result = self.compiled_model([padded])
        out = np.asarray(result[0])
        slicer = [slice(None)] * out.ndim
        slicer[self.axis] = slice(0, real_len)
        return {0: out[tuple(slicer)]}


def load_component(
    core: ov.Core,
    xml_path: str,
    component: str,
    device: str = "CPU",
    static_shape=None,
    npu_config: dict = None,
    fallback_device: str = "CPU",
):
    """Load+compile a single OpenVINO IR component with explicit device
    selection, optional static-shape reshape (required for NPU), and
    automatic fallback to `fallback_device` if compiling/reshaping on the
    requested device raises (unsupported op, missing static shape, device
    not present, etc.).

    Returns `(model_callable, device_actually_used)`. `model_callable` is
    compatible with how ov_cosyvoice_helper.py calls its compiled models
    (`result = model(inputs); result[0]` for the first output) -- either a
    raw `ov.CompiledModel` or a `PaddedCompiledModel` wrapper for the two
    verified pad-safe components.
    """
    npu_config = npu_config or {}
    device = (device or "CPU").upper()

    def _compile(dev, reshape_to):
        model = core.read_model(xml_path)
        if reshape_to is not None:
            model.reshape(reshape_to)
        compiled = core.compile_model(model, dev, npu_config if dev == "NPU" else {})
        return compiled

    if device == "NPU":
        if static_shape is None:
            warnings.warn(
                f"[{component}] NPU requires a static input shape but none "
                f"was configured -- falling back to {fallback_device}."
            )
            return _compile(fallback_device, None), fallback_device
        try:
            compiled = _compile("NPU", static_shape)
        except Exception as e:  # noqa: BLE001 - intentionally broad: any NPU compile failure should fall back
            warnings.warn(
                f"[{component}] failed to compile on NPU with static shape "
                f"{static_shape} ({e!r}) -- falling back to {fallback_device}."
            )
            return _compile(fallback_device, None), fallback_device

        if component in PAD_SAFE_AXIS:
            axis = PAD_SAFE_AXIS[component]
            return PaddedCompiledModel(compiled, static_shape[axis], axis), "NPU"

        print(
            f"[{component}] compiled on NPU with static shape {static_shape} "
            f"-- EXACT match required at inference time, padding is not "
            f"applied for this component (see device_loader.py docstring)."
        )
        return compiled, "NPU"

    try:
        return _compile(device, None), device
    except Exception as e:  # noqa: BLE001
        if device == fallback_device:
            raise
        warnings.warn(f"[{component}] failed to compile on {device} ({e!r}) -- falling back to {fallback_device}.")
        return _compile(fallback_device, None), fallback_device
