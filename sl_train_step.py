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


def _get_compiled_forward_with_teacher(model):
    """Compile teacher + student forward as one unit for better fusion."""
    compiled = getattr(model, "_compiled_fwd_with_teacher", None)
    if compiled is None:
        def _fwd_with_teacher(batch):
            with torch.no_grad():
                teacher_targets = model.compute_teacher_targets(batch)
            return model.forward_augmented(batch, teacher_targets)
        compiled = torch.compile(_fwd_with_teacher, mode="max-autotune")
        model._compiled_fwd_with_teacher = compiled
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


def _get_trainable_params(model):
    params = getattr(model, "_trainable_params", None)
    if params is None:
        params = [param for param in model.parameters() if param.requires_grad]
        model._trainable_params = params
    return params


def train_step(model, batch, optimizer, autocast_dtype, grad_clip_norm):
    """One complete training step. Returns metrics dict."""
    model.advance_sigreg_lambda_schedule()

    # Combined teacher + student forward (single compiled graph)
    torch.compiler.cudagraph_mark_step_begin()
    if model.teacher_encoder is not None:
        compiled_fwd = _get_compiled_forward_with_teacher(model)
        with torch.autocast("cuda", dtype=autocast_dtype):
            metrics = compiled_fwd(batch)
    else:
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
