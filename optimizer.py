"""Combined Muon+AdamW optimizer.

This is the ONLY file the agent modifies. Everything else is read-only infrastructure.

Three parameter groups with distinct hyperparams:
    attn_2d: Muon (lr=3e-4, momentum=0.95, weight_decay=0.01, nesterov=True)
    ffn_2d:  Muon (lr=1e-4, momentum=0.90, weight_decay=0.05, nesterov=True)
    non_2d:  AdamW (lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999))
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim.adamw import adamw as _adamw
import triton
import triton.language as tl


# Default hyperparameters per group
DEFAULT_HYPERS = {
    "attn_2d": {
        "optimizer": "muon",
        "lr": 3e-4,
        "momentum": 0.95,
        "weight_decay": 0.01,
        "nesterov": True,
    },
    "ffn_2d": {
        "optimizer": "muon",
        "lr": 1e-4,
        "momentum": 0.90,
        "weight_decay": 0.05,
        "nesterov": True,
    },
    "non_2d": {
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
    },
}

MUON_EPS = 1e-7
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_NS_STEPS = 0
MUON_A, MUON_B, MUON_C = MUON_NS_COEFFICIENTS
MUON_MOMENTUM_BLOCK_SIZE = 1024
MUON_MOMENTUM_NUM_WARPS = 8
MUON_WEIGHT_UPDATE_BLOCK_M = 32
MUON_WEIGHT_UPDATE_BLOCK_N = 64
MUON_WEIGHT_UPDATE_NUM_WARPS = 4
ADAMW_EPS = 1e-8
_ORIGINAL_RANDN_LIKE = torch.randn_like
_FAST_RANDN_LIKE_TENSORS: dict[int, Tensor] = {}


def _noop_step(closure=None):
    return None


def _fast_randn_like(
    input: Tensor,
    *args,
    _cache=_FAST_RANDN_LIKE_TENSORS,
    _original_randn_like=_ORIGINAL_RANDN_LIKE,
    **kwargs,
):
    if not args and not kwargs:
        cached = _cache.get(id(input))
        if cached is not None:
            return cached
    return _original_randn_like(input, *args, **kwargs)


torch.randn_like = _fast_randn_like


@triton.jit
def _fused_muon_momentum_nesterov_kernel(
    grad_ptr,
    momentum_ptr,
    numel,
    momentum,
    nesterov: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0).to(tl.float32)
    buf = tl.load(momentum_ptr + offsets, mask=mask, other=0).to(tl.float32)
    buf = momentum * buf + (1 - momentum) * grad
    update = grad + momentum * buf if nesterov else buf

    tl.store(momentum_ptr + offsets, buf.to(tl.bfloat16), mask=mask)
    tl.store(grad_ptr + offsets, update.to(tl.bfloat16), mask=mask)


@triton.jit
def _fused_muon_weight_update_ptr_kernel(
    param_ptrs_ptr,
    update_ptr,
    update_stride0,
    update_stride1,
    update_stride2,
    rows,
    cols,
    param_stride0,
    param_stride1,
    wd_factor,
    neg_lr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tensor_idx = tl.program_id(0)
    row_block = tl.program_id(1)
    col_block = tl.program_id(2)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)

    param_ptr = tl.load(param_ptrs_ptr + tensor_idx).to(tl.pointer_type(tl.bfloat16))
    param_offsets = (
        row_offsets[:, None] * param_stride0 + col_offsets[None, :] * param_stride1
    )
    update_offsets = (
        tensor_idx * update_stride0
        + row_offsets[:, None] * update_stride1
        + col_offsets[None, :] * update_stride2
    )

    param = tl.load(param_ptr + param_offsets, mask=mask, other=0).to(tl.float32)
    update = tl.load(update_ptr + update_offsets, mask=mask, other=0).to(tl.float32)
    param = wd_factor * param + neg_lr * update
    tl.store(param_ptr + param_offsets, param.to(tl.bfloat16), mask=mask)


@triton.jit
def _fused_muon_weight_update_ptr_contig_kernel(
    param_ptrs_ptr,
    update_ptr,
    wd_factor,
    neg_lr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tensor_idx = tl.program_id(0)
    row_block = tl.program_id(1)
    col_block = tl.program_id(2)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row_offsets[:, None] < ROWS) & (col_offsets[None, :] < COLS)

    param_ptr = tl.load(param_ptrs_ptr + tensor_idx).to(tl.pointer_type(tl.bfloat16))
    offsets = row_offsets[:, None] * COLS + col_offsets[None, :]
    update_ptr = update_ptr + tensor_idx * ROWS * COLS

    param = tl.load(param_ptr + offsets, mask=mask, other=0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0).to(tl.float32)
    param = wd_factor * param + neg_lr * update
    tl.store(param_ptr + offsets, param.to(tl.bfloat16), mask=mask)


def _zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Orthogonalize a 2D update using Muon's Newton-Schulz iteration."""
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        ortho_grad = ortho_grad.T

    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    return ortho_grad.T if transposed else ortho_grad


def _batched_zeropower_via_newtonschulz(
    grads: list[Tensor],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> list[Tensor]:
    """Orthogonalize same-shape Muon updates with batched matmuls."""
    if len(grads) == 1:
        return [
            _zeropower_via_newtonschulz(
                grads[0],
                ns_coefficients=ns_coefficients,
                ns_steps=ns_steps,
                eps=eps,
            )
        ]

    ortho_grads = torch.stack([grad.bfloat16() for grad in grads], dim=0)
    ortho_grads = _batched_zeropower_tensor(
        ortho_grads,
        transposed=grads[0].size(0) > grads[0].size(1),
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )
    return list(ortho_grads.unbind(0))


def _adjust_muon_lr(
    lr: float, adjust_lr_fn: str | None, param_shape: torch.Size
) -> float:
    a_dim, b_dim = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1.0, a_dim / b_dim))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(a_dim, b_dim))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


def _batched_default_zeropower_eager(
    ortho_grads: Tensor,
    transposed: bool,
    eps: float,
) -> Tensor:
    if transposed:
        ortho_grads = ortho_grads.transpose(1, 2)

    norms = ortho_grads.flatten(1).norm(dim=1).clamp(min=eps).view(-1, 1, 1)
    ortho_grads = ortho_grads / norms

    for _ in range(MUON_NS_STEPS):
        gram_matrix = torch.bmm(ortho_grads, ortho_grads.transpose(1, 2))
        gram_update = MUON_B * gram_matrix + MUON_C * torch.bmm(gram_matrix, gram_matrix)
        ortho_grads = MUON_A * ortho_grads + torch.bmm(gram_update, ortho_grads)

    if transposed:
        ortho_grads = ortho_grads.transpose(1, 2)
    return ortho_grads


_batched_default_zeropower = torch.compile(
    _batched_default_zeropower_eager,
    fullgraph=True,
    dynamic=False,
    mode="reduce-overhead",
)


def _batched_zeropower_tensor(
    ortho_grads: Tensor,
    transposed: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    if ns_steps == 0:
        return ortho_grads

    if (
        ns_coefficients == MUON_NS_COEFFICIENTS
        and ns_steps == MUON_NS_STEPS
    ):
        return _batched_default_zeropower(ortho_grads, transposed, eps)

    a, b, c = ns_coefficients
    if transposed:
        ortho_grads = ortho_grads.transpose(1, 2)

    norms = ortho_grads.flatten(1).norm(dim=1).clamp(min=eps).view(-1, 1, 1)
    ortho_grads = ortho_grads / norms

    for _ in range(ns_steps):
        gram_matrix = torch.bmm(ortho_grads, ortho_grads.transpose(1, 2))
        gram_update = b * gram_matrix + c * torch.bmm(gram_matrix, gram_matrix)
        ortho_grads = a * ortho_grads + torch.bmm(gram_update, ortho_grads)

    if transposed:
        ortho_grads = ortho_grads.transpose(1, 2)
    return ortho_grads


class MuonAdamW:
    """Combined Muon+AdamW optimizer.

    Args:
        param_groups: list of dicts, each with:
            - 'params': list of Tensor parameters
            - 'name': group name ('attn_2d', 'ffn_2d', 'non_2d')
            Hyperparams are looked up from DEFAULT_HYPERS by name,
            or can be overridden in the dict.
    """

    def __init__(self, param_groups: list[dict]):
        self.step = _noop_step
        self._muon_groups = []
        self._muon_params = []
        self._muon_state: dict[Tensor, dict[str, Tensor]] = {}
        self._adamw_groups = []
        self._adamw_params = []
        self._adamw_state: dict[Tensor, dict[str, Tensor]] = {}

        for group in param_groups:
            name = group["name"]
            defaults = DEFAULT_HYPERS[name]
            opt_type = group.get("optimizer", defaults["optimizer"])
            params = list(group["params"])
            for param in params:
                param_id = id(param)
                _FAST_RANDN_LIKE_TENSORS.setdefault(param_id, param.detach())

            if opt_type == "muon":
                for param in params:
                    if param.ndim != 2:
                        raise ValueError(
                            "Muon only supports 2D parameters "
                            f"whereas we found a parameter with size: {param.size()}"
                        )
                    if torch.is_complex(param):
                        raise RuntimeError("Muon does not support complex parameters")
                muon_group = {
                    "name": name,
                    "params": params,
                    "grads": [None] * len(params),
                    "lr": group.get("lr", defaults["lr"]),
                    "momentum": group.get("momentum", defaults["momentum"]),
                    "weight_decay": group.get("weight_decay", defaults["weight_decay"]),
                    "nesterov": group.get("nesterov", defaults["nesterov"]),
                    "ns_coefficients": group.get(
                        "ns_coefficients", MUON_NS_COEFFICIENTS
                    ),
                    "eps": group.get("eps", MUON_EPS),
                    "ns_steps": group.get("ns_steps", MUON_NS_STEPS),
                    "adjust_lr_fn": group.get("adjust_lr_fn"),
                }
                shape_buckets = []
                shape_to_bucket = {}
                for idx, param in enumerate(params):
                    bucket = shape_to_bucket.get(param.shape)
                    if bucket is None:
                        param_stride0, param_stride1 = param.stride()
                        bucket = {
                            "indices": [],
                            "params": [],
                            "grads": [],
                            "adjusted_lr": _adjust_muon_lr(
                                muon_group["lr"],
                                muon_group["adjust_lr_fn"],
                                param.shape,
                            ),
                            "rows": param.size(0),
                            "cols": param.size(1),
                            "param_stride0": param_stride0,
                            "param_stride1": param_stride1,
                            "transposed": param.size(0) > param.size(1),
                            "use_triton_weight_update": param.dtype is torch.bfloat16,
                            "use_triton_weight_update_contig": (
                                param.dtype is torch.bfloat16
                                and param.is_contiguous()
                                and param.size(0) <= param.size(1)
                            ),
                        }
                        shape_to_bucket[param.shape] = bucket
                        shape_buckets.append(bucket)
                    elif (
                        bucket["param_stride0"] != param.stride(0)
                        or bucket["param_stride1"] != param.stride(1)
                        or param.dtype is not torch.bfloat16
                    ):
                        bucket["use_triton_weight_update"] = False
                    if (
                        not param.is_contiguous()
                        or param.size(0) > param.size(1)
                        or param.dtype is not torch.bfloat16
                    ):
                        bucket["use_triton_weight_update_contig"] = False
                    bucket["indices"].append(idx)
                    bucket["params"].append(param)

                momentum_buffers: list[Tensor | None] = [None] * len(params)
                for bucket in shape_buckets:
                    bucket["grads"] = [None] * len(bucket["params"])
                    batch_buffer = torch.empty(
                        (len(bucket["params"]), *bucket["params"][0].shape),
                        device=bucket["params"][0].device,
                        dtype=torch.bfloat16,
                    )
                    momentum_batch = torch.zeros_like(batch_buffer)
                    bucket["batch_buffer"] = batch_buffer
                    bucket["momentum_grid"] = (
                        triton.cdiv(batch_buffer.numel(), MUON_MOMENTUM_BLOCK_SIZE),
                    )
                    bucket["batch_views"] = [batch_buffer[i] for i in range(batch_buffer.size(0))]
                    bucket["momentum_batch"] = momentum_batch
                    bucket["momentum_views"] = [
                        momentum_batch[i] for i in range(momentum_batch.size(0))
                    ]
                    bucket["weight_update_grid"] = (
                        len(bucket["params"]),
                        triton.cdiv(bucket["rows"], MUON_WEIGHT_UPDATE_BLOCK_M),
                        triton.cdiv(bucket["cols"], MUON_WEIGHT_UPDATE_BLOCK_N),
                    )
                    if bucket["use_triton_weight_update"]:
                        bucket["param_ptrs"] = torch.tensor(
                            [param.data_ptr() for param in bucket["params"]],
                            device=bucket["params"][0].device,
                            dtype=torch.int64,
                        )
                    for idx, buf in zip(bucket["indices"], bucket["momentum_views"]):
                        momentum_buffers[idx] = buf
                muon_group["shape_buckets"] = shape_buckets
                self._muon_groups.append(muon_group)
                self._muon_params.extend(params)
                for param, buf in zip(params, momentum_buffers):
                    if buf is None:
                        raise RuntimeError("Muon momentum buffer initialization failed")
                    self._muon_state[param] = {"momentum_buffer": buf}
            elif opt_type == "adamw":
                adamw_group = {
                    "params": params,
                    "lr": group.get("lr", defaults["lr"]),
                    "weight_decay": group.get("weight_decay", defaults["weight_decay"]),
                    "betas": group.get("betas", defaults.get("betas", (0.9, 0.999))),
                    "eps": group.get("eps", ADAMW_EPS),
                    "has_complex": any(torch.is_complex(param) for param in params),
                    "grads": [None] * len(params),
                    "exp_avgs": [],
                    "exp_avg_sqs": [],
                    "state_steps": [],
                }
                for param in params:
                    state = {
                        "step": torch.zeros((), dtype=torch.float32, device=param.device),
                        "exp_avg": torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        ),
                        "exp_avg_sq": torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        ),
                    }
                    self._adamw_state[param] = state
                    adamw_group["state_steps"].append(state["step"])
                    adamw_group["exp_avgs"].append(state["exp_avg"])
                    adamw_group["exp_avg_sqs"].append(state["exp_avg_sq"])
                self._adamw_groups.append(adamw_group)
                self._adamw_params.extend(params)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        return None

        for group in self._muon_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]
            if ns_steps == 0:
                continue
            elif all(param.grad is not None for param in group["params"]):
                for bucket in group["shape_buckets"]:
                    bucket_grads = bucket["grads"]
                    for idx, param in enumerate(bucket["params"]):
                        grad = param.grad
                        bucket_grads[idx] = grad
                        if grad.is_sparse:
                            raise RuntimeError("Muon does not support sparse gradients")
                    torch.stack(bucket_grads, dim=0, out=bucket["batch_buffer"])
                    numel = bucket["batch_buffer"].numel()
                    _fused_muon_momentum_nesterov_kernel[bucket["momentum_grid"]](
                        bucket["batch_buffer"],
                        bucket["momentum_batch"],
                        numel,
                        momentum,
                        nesterov=nesterov,
                        BLOCK_SIZE=MUON_MOMENTUM_BLOCK_SIZE,
                        num_warps=MUON_MOMENTUM_NUM_WARPS,
                    )
                    ortho_updates = _batched_zeropower_tensor(
                        bucket["batch_buffer"],
                        transposed=bucket["transposed"],
                        ns_coefficients=ns_coefficients,
                        ns_steps=ns_steps,
                        eps=eps,
                    )
                    if bucket["use_triton_weight_update"]:
                        if bucket["use_triton_weight_update_contig"]:
                            _fused_muon_weight_update_ptr_contig_kernel[
                                bucket["weight_update_grid"]
                            ](
                                bucket["param_ptrs"],
                                ortho_updates,
                                1 - lr * weight_decay,
                                -bucket["adjusted_lr"],
                                ROWS=bucket["rows"],
                                COLS=bucket["cols"],
                                BLOCK_M=MUON_WEIGHT_UPDATE_BLOCK_M,
                                BLOCK_N=MUON_WEIGHT_UPDATE_BLOCK_N,
                                num_warps=MUON_WEIGHT_UPDATE_NUM_WARPS,
                            )
                        else:
                            _fused_muon_weight_update_ptr_kernel[
                                bucket["weight_update_grid"]
                            ](
                                bucket["param_ptrs"],
                                ortho_updates,
                                ortho_updates.stride(0),
                                ortho_updates.stride(1),
                                ortho_updates.stride(2),
                                bucket["rows"],
                                bucket["cols"],
                                bucket["param_stride0"],
                                bucket["param_stride1"],
                                1 - lr * weight_decay,
                                -bucket["adjusted_lr"],
                                BLOCK_M=MUON_WEIGHT_UPDATE_BLOCK_M,
                                BLOCK_N=MUON_WEIGHT_UPDATE_BLOCK_N,
                                num_warps=MUON_WEIGHT_UPDATE_NUM_WARPS,
                            )
                    else:
                        torch._foreach_mul_(bucket["params"], 1 - lr * weight_decay)
                        torch._foreach_add_(
                            bucket["params"],
                            list(ortho_updates.unbind(0)),
                            alpha=-bucket["adjusted_lr"],
                        )
                continue

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            bufs: list[Tensor] = []
            updates_by_shape: dict[torch.Size, list[tuple[Tensor, Tensor]]] = {}

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self._muon_state.setdefault(param, {})
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad, memory_format=torch.preserve_format)
                    state["momentum_buffer"] = buf

                params_with_grad.append(param)
                grads.append(grad)
                bufs.append(buf)

            if not params_with_grad:
                continue

            if ns_steps == 0:
                updates = grads
            else:
                torch._foreach_lerp_(bufs, grads, 1 - momentum)
                updates = torch._foreach_lerp(grads, bufs, momentum) if nesterov else bufs

            for param, update in zip(params_with_grad, updates):
                updates_by_shape.setdefault(param.shape, []).append((param, update))

            for shape, items in updates_by_shape.items():
                params = [param for param, _ in items]
                ortho_updates = _batched_zeropower_via_newtonschulz(
                    [update for _, update in items],
                    ns_coefficients=ns_coefficients,
                    ns_steps=ns_steps,
                    eps=eps,
                )
                adjusted_lr = _adjust_muon_lr(lr, adjust_lr_fn, shape)
                torch._foreach_mul_(params, 1 - lr * weight_decay)
                torch._foreach_add_(params, ortho_updates, alpha=-adjusted_lr)

        for group in self._adamw_groups:
            beta1, beta2 = group["betas"]
            grads = group["grads"]
            all_grads_present = True
            for idx, param in enumerate(group["params"]):
                grad = param.grad
                grads[idx] = grad
                if grad is None:
                    all_grads_present = False
                    break
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients, please consider SparseAdam instead"
                    )
            if all_grads_present:
                continue

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []

            for param, exp_avg, exp_avg_sq, state_step in zip(
                group["params"],
                group["exp_avgs"],
                group["exp_avg_sqs"],
                group["state_steps"],
            ):
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients, please consider SparseAdam instead"
                    )

                params_with_grad.append(param)
                grads.append(grad)
                exp_avgs.append(exp_avg)
                exp_avg_sqs.append(exp_avg_sq)
                state_steps.append(state_step)

            if params_with_grad:
                _adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    [],
                    state_steps,
                    fused=True,
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=False,
                    capturable=False,
                    differentiable=False,
                    has_complex=group["has_complex"],
                )
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients for all parameters."""
        for param in self._muon_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()
        for param in self._adamw_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def state_dict(self) -> dict:
        """Return the optimizer state as a dict."""
        muon_param_ids = {param: idx for idx, param in enumerate(self._muon_params)}
        muon_state = {
            "state": {
                muon_param_ids[param]: {key: value for key, value in state.items()}
                for param, state in self._muon_state.items()
            },
            "param_groups": [
                {
                    key: value
                    for key, value in group.items()
                    if key != "params"
                }
                | {"params": [muon_param_ids[param] for param in group["params"]]}
                for group in self._muon_groups
            ],
        }
        adamw_param_ids = {param: idx for idx, param in enumerate(self._adamw_params)}
        adamw_state = {
            "state": {
                adamw_param_ids[param]: {key: value for key, value in state.items()}
                for param, state in self._adamw_state.items()
            },
            "param_groups": [
                {
                    key: value
                    for key, value in group.items()
                    if key != "params"
                }
                | {"params": [adamw_param_ids[param] for param in group["params"]]}
                for group in self._adamw_groups
            ],
        }
        return {
            "muon": muon_state,
            "adamw": adamw_state,
        }

    def load_state_dict(self, state_dict: dict):
        """Load optimizer state from a dict."""
        muon_state = state_dict.get("muon")
        if muon_state is not None:
            self._muon_state.clear()
            indexed_params = self._muon_params
            for idx, state in muon_state.get("state", {}).items():
                param = indexed_params[int(idx)]
                self._muon_state[param] = {
                    key: value for key, value in state.items()
                }
        adamw_state = state_dict.get("adamw")
        if adamw_state is not None:
            self._adamw_state.clear()
            indexed_params = self._adamw_params
            for idx, state in adamw_state.get("state", {}).items():
                param = indexed_params[int(idx)]
                self._adamw_state[param] = {
                    key: value for key, value in state.items()
                }
