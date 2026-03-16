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


def _get_compiled_forward_augmented(model):
    compiled = getattr(model, "_compiled_forward_augmented", None)
    if compiled is None:
        compiled = torch.compile(
            model.forward_augmented,
            mode="max-autotune",
            dynamic=False,
        )
        model._compiled_forward_augmented = compiled
    return compiled


def _get_compiled_context_encoder(model):
    compiled = getattr(model, "_compiled_context_encoder", None)
    if compiled is None:
        compiled = torch.compile(
            model.compute_context_emb,
            mode="max-autotune-no-cudagraphs",
            dynamic=False,
        )
        model._compiled_context_encoder = compiled
    return compiled


def _get_compiled_predictor_loss(model):
    compiled = getattr(model, "_compiled_predictor_loss", None)
    if compiled is None:
        compiled = torch.compile(
            model.forward_predictor_loss,
            mode="max-autotune-no-cudagraphs",
            dynamic=False,
        )
        model._compiled_predictor_loss = compiled
    return compiled


def _get_teacher_runner(model):
    runner = getattr(model, "_teacher_graph_runner", None)
    if runner is None:
        runner = _CUDAGraphRunner(compile_kwargs={"mode": "max-autotune-no-cudagraphs", "dynamic": False})
        model._teacher_graph_runner = runner
    return runner


def _get_teacher_stream():
    stream = getattr(_get_teacher_stream, "_stream", None)
    if stream is None:
        stream = torch.cuda.Stream()
        _get_teacher_stream._stream = stream
    return stream


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
        # Overlap teacher CUDA graph with student context encoder
        teacher_stream = _get_teacher_stream()
        runner = _get_teacher_runner(model)
        main_stream = torch.cuda.current_stream()

        # Launch teacher on side stream
        teacher_stream.wait_stream(main_stream)
        with torch.cuda.stream(teacher_stream):
            with torch.autocast("cuda", dtype=autocast_dtype):
                teacher_targets = runner.run(model.compute_teacher_targets, batch)

        # Run context encoder on main stream (overlapped with teacher)
        compiled_encoder = _get_compiled_context_encoder(model)
        with torch.autocast("cuda", dtype=autocast_dtype):
            context_emb = compiled_encoder(batch)

        # Wait for teacher to finish before predictor+loss
        main_stream.wait_stream(teacher_stream)

        # Run predictor + loss on main stream
        compiled_pred_loss = _get_compiled_predictor_loss(model)
        with torch.autocast("cuda", dtype=autocast_dtype):
            metrics = compiled_pred_loss(batch, context_emb, teacher_targets)
    else:
        # No teacher — use original single-compile path
        torch.compiler.cudagraph_mark_step_begin()
        forward_augmented = _get_compiled_forward_augmented(model)
        with torch.autocast("cuda", dtype=autocast_dtype):
            metrics = forward_augmented(batch, None)

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
