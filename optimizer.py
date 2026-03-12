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
MUON_NS_STEPS = 5
MUON_A, MUON_B, MUON_C = MUON_NS_COEFFICIENTS


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
    transposed = grads[0].size(0) > grads[0].size(1)
    if (
        ns_coefficients == MUON_NS_COEFFICIENTS
        and ns_steps == MUON_NS_STEPS
    ):
        ortho_grads = _batched_default_zeropower(ortho_grads, transposed, eps)
    else:
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
        self._muon_groups = []
        self._muon_params = []
        self._muon_state: dict[Tensor, dict[str, Tensor]] = {}
        adamw_groups = []

        for group in param_groups:
            name = group["name"]
            defaults = DEFAULT_HYPERS[name]
            opt_type = group.get("optimizer", defaults["optimizer"])
            params = list(group["params"])

            if opt_type == "muon":
                for param in params:
                    if param.ndim != 2:
                        raise ValueError(
                            "Muon only supports 2D parameters "
                            f"whereas we found a parameter with size: {param.size()}"
                        )
                muon_group = {
                    "params": params,
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
                self._muon_groups.append(muon_group)
                self._muon_params.extend(params)
            elif opt_type == "adamw":
                adamw_groups.append({
                    "params": params,
                    "lr": group.get("lr", defaults["lr"]),
                    "weight_decay": group.get("weight_decay", defaults["weight_decay"]),
                    "betas": group.get("betas", defaults.get("betas", (0.9, 0.999))),
                })

        self._adamw = torch.optim.AdamW(adamw_groups) if adamw_groups else None

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self._muon_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            bufs: list[Tensor] = []
            updates_by_shape: dict[torch.Size, list[tuple[Tensor, Tensor]]] = {}

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if torch.is_complex(param):
                    raise RuntimeError("Muon does not support complex parameters")
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

        if self._adamw is not None:
            self._adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients for all parameters."""
        for param in self._muon_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)

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
        return {
            "muon": muon_state,
            "adamw": self._adamw.state_dict() if self._adamw is not None else None,
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
        if self._adamw is not None and state_dict.get("adamw") is not None:
            self._adamw.load_state_dict(state_dict["adamw"])
