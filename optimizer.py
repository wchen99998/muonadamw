"""Combined Muon+AdamW optimizer.

This is the ONLY file the agent modifies. Everything else is read-only infrastructure.

Three parameter groups with distinct hyperparams:
    attn_2d: Muon (lr=3e-4, momentum=0.95, weight_decay=0.01, nesterov=True)
    ffn_2d:  Muon (lr=1e-4, momentum=0.90, weight_decay=0.05, nesterov=True)
    non_2d:  AdamW (lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999))
"""

from __future__ import annotations

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
        muon_groups = []
        adamw_groups = []

        for group in param_groups:
            name = group["name"]
            defaults = DEFAULT_HYPERS[name]
            opt_type = group.get("optimizer", defaults["optimizer"])

            if opt_type == "muon":
                muon_groups.append({
                    "params": group["params"],
                    "lr": group.get("lr", defaults["lr"]),
                    "momentum": group.get("momentum", defaults["momentum"]),
                    "weight_decay": group.get("weight_decay", defaults["weight_decay"]),
                    "nesterov": group.get("nesterov", defaults["nesterov"]),
                })
            elif opt_type == "adamw":
                adamw_groups.append({
                    "params": group["params"],
                    "lr": group.get("lr", defaults["lr"]),
                    "weight_decay": group.get("weight_decay", defaults["weight_decay"]),
                    "betas": group.get("betas", defaults.get("betas", (0.9, 0.999))),
                })

        self._muon = torch.optim.Muon(muon_groups) if muon_groups else None
        self._adamw = torch.optim.AdamW(adamw_groups) if adamw_groups else None

    def step(self, closure=None):
        """Perform a single optimization step."""
        if self._muon is not None:
            self._muon.step()
        if self._adamw is not None:
            self._adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients for all parameters."""
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        """Return the optimizer state as a dict."""
        return {
            "muon": self._muon.state_dict() if self._muon is not None else None,
            "adamw": self._adamw.state_dict() if self._adamw is not None else None,
        }

    def load_state_dict(self, state_dict: dict):
        """Load optimizer state from a dict."""
        if self._muon is not None and state_dict.get("muon") is not None:
            self._muon.load_state_dict(state_dict["muon"])
        if self._adamw is not None and state_dict.get("adamw") is not None:
            self._adamw.load_state_dict(state_dict["adamw"])
