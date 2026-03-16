"""Optimizable training step for PeakSetSIGReg.

This file handles step-level optimizations: compilation, CUDA graphs,
compute/memory overlap. Model-level optimizations go in sl_model_opt.py.
Together they form the complete optimization target.
"""

import torch


class _CUDAGraphRunner:
    """Manages an explicit CUDA graph for a compiled function.

    Uses manual CUDA graph capture to avoid torch.compile CUDAGraphTree
    overhead (clone, tree management).
    """

    def __init__(self, compile_kwargs=None):
        self.graph = None
        self.static_output = None
        self.compiled_fn = None
        self.compile_kwargs = compile_kwargs or {}

    def _warmup_and_capture(self, fn, *args):
        if self.compiled_fn is None:
            self.compiled_fn = torch.compile(fn, **self.compile_kwargs)

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.compiled_fn(*args)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.compiled_fn(*args)

    def run(self, fn, *args):
        if self.graph is None:
            self._warmup_and_capture(fn, *args)
        self.graph.replay()
        return self.static_output


def _get_compiled_encode_context(model):
    compiled = getattr(model, "_compiled_encode_context", None)
    if compiled is None:
        compiled = torch.compile(
            model.encode_context,
            mode="max-autotune",
        )
        model._compiled_encode_context = compiled
    return compiled


def _get_compiled_forward_augmented(model):
    compiled = getattr(model, "_compiled_forward_augmented", None)
    if compiled is None:
        compiled = torch.compile(
            model.forward_augmented,
            mode="max-autotune",
        )
        model._compiled_forward_augmented = compiled
    return compiled


def _get_teacher_runner(model):
    runner = getattr(model, "_teacher_graph_runner", None)
    if runner is None:
        runner = _CUDAGraphRunner(compile_kwargs={"mode": "max-autotune-no-cudagraphs"})
        model._teacher_graph_runner = runner
    return runner


def _get_teacher_stream():
    """Get or create a persistent CUDA stream for teacher computation."""
    if not hasattr(_get_teacher_stream, "_stream"):
        _get_teacher_stream._stream = torch.cuda.Stream()
    return _get_teacher_stream._stream


def _get_trainable_params(model):
    params = getattr(model, "_trainable_params", None)
    if params is None:
        params = [param for param in model.parameters() if param.requires_grad]
        model._trainable_params = params
    return params


def train_step(model, batch, optimizer, autocast_dtype, grad_clip_norm):
    """One complete training step. Returns metrics dict."""
    model.advance_sigreg_lambda_schedule()

    if model.teacher_encoder is not None:
        runner = _get_teacher_runner(model)
        teacher_stream = _get_teacher_stream()
        current_stream = torch.cuda.current_stream()

        # Ensure teacher graph is initialized (first call only)
        if runner.graph is None:
            with torch.autocast("cuda", dtype=autocast_dtype):
                runner.run(model.compute_teacher_targets, batch)

        # Launch teacher CUDA graph replay on separate stream
        teacher_stream.wait_stream(current_stream)
        with torch.cuda.stream(teacher_stream):
            runner.graph.replay()
        teacher_targets = runner.static_output

        # Student encoder on default stream (overlaps with teacher on teacher_stream)
        torch.compiler.cudagraph_mark_step_begin()
        encode_context = _get_compiled_encode_context(model)
        with torch.autocast("cuda", dtype=autocast_dtype):
            context_emb = encode_context(batch)

        # Wait for teacher before predictor+loss (which needs teacher_targets)
        current_stream.wait_stream(teacher_stream)

        # Predictor + loss using pre-computed context_emb and teacher_targets
        forward_augmented = _get_compiled_forward_augmented(model)
        with torch.autocast("cuda", dtype=autocast_dtype):
            metrics = forward_augmented(batch, teacher_targets, context_emb)
    else:
        torch.compiler.cudagraph_mark_step_begin()
        forward_augmented = _get_compiled_forward_augmented(model)
        with torch.autocast("cuda", dtype=autocast_dtype):
            metrics = forward_augmented(batch)

    metrics["loss"].backward()
    if grad_clip_norm and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            _get_trainable_params(model),
            max_norm=grad_clip_norm,
        )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.update_teacher()
    return metrics
