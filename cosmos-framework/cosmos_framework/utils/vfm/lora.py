# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Custom in-place LoRA injection for MoT-style models.

The key design choice is to subclass ``nn.Linear`` (``LoraInjectedLinear``)
rather than wrap it (as PEFT's ``LoraLayer`` does). The wrapped weight stays
at ``<path>.weight`` so checkpoints saved from a LoRA-trained model load
cleanly into either a LoRA or non-LoRA model — no key rename and no loader
alias needed. ``lora_A`` and ``lora_B`` are sibling submodules, producing
the state-dict keys ``<path>.lora_A.weight`` and ``<path>.lora_B.weight``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.utils import log


def load_lora_weights(model: nn.Module, lora_path: str | Path) -> int:
    """Load an exported standalone Reasoner LoRA into a native Cosmos model.

    The exporter stores complete ``language_model.*.lora_[AB].weight`` tensors,
    while a distributed model may expose FSDP/DTensor-backed parameters.  The
    distributed state-dict API handles scattering the partial adapter state;
    plain ``load_state_dict`` remains the fallback for CPU/test models.
    """

    import safetensors.torch
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
    from torch.distributed.tensor import DTensor

    root = Path(lora_path).expanduser()
    if root.is_file():
        files = [root]
    elif root.is_dir():
        canonical = root / "reasoner_lora.safetensors"
        preferred = root / "adapter_model.safetensors"
        if canonical.is_file():
            files = [canonical]
        elif preferred.is_file():
            files = [preferred]
        else:
            files = sorted(root.glob("*.safetensors"))
    else:
        raise FileNotFoundError(f"Reasoner LoRA path does not exist: {root}")
    if not files:
        raise FileNotFoundError(f"No safetensors files found under Reasoner LoRA path: {root}")

    source: dict[str, torch.Tensor] = {}
    for file_path in files:
        source.update(safetensors.torch.load_file(str(file_path), device="cpu"))
    source = {key: value for key, value in source.items() if "lora_" in key}
    if not source:
        raise ValueError(f"Reasoner LoRA artifact contains no lora_ tensors: {root}")

    try:
        target_state = get_model_state_dict(model)
    except Exception:
        target_state = model.state_dict()
    target_is_distributed = any(isinstance(value, DTensor) for value in target_state.values())
    target_keys = set(target_state)
    mapped: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []
    for source_key, value in source.items():
        # Standalone exports use the Omni namespace
        # ``language_model.model.layers...``.  The pure Qwen VLM passed here
        # exposes the same modules as ``model.language_model.layers...``;
        # retain both layouts so exported adapters can initialize continuation
        # runs as well as the network layout used during export.
        candidates: list[str] = []

        def add_candidate(candidate: str) -> None:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add_candidate(source_key)
        add_candidate(source_key.removeprefix("model."))
        add_candidate(source_key.removeprefix("module."))
        add_candidate(source_key.removeprefix("model.").removeprefix("module."))
        for candidate in tuple(candidates):
            if candidate.startswith("language_model.model."):
                inner = candidate.removeprefix("language_model.model.")
                add_candidate(f"model.language_model.{inner}")
                add_candidate(f"language_model.{inner}")
            elif candidate.startswith("language_model."):
                inner = candidate.removeprefix("language_model.")
                add_candidate(f"model.language_model.{inner}")
            if candidate.startswith("model.language_model.model."):
                inner = candidate.removeprefix("model.language_model.model.")
                add_candidate(f"model.language_model.{inner}")
        target_key = next((candidate for candidate in candidates if candidate in target_keys), None)
        if target_key is None:
            suffix_matches = [key for key in target_keys if key.endswith(candidates[1])]
            if len(suffix_matches) == 1:
                target_key = suffix_matches[0]
        if target_key is None:
            unmatched.append(source_key)
            continue
        if tuple(target_state[target_key].shape) != tuple(value.shape):
            raise ValueError(
                f"Reasoner LoRA tensor shape mismatch for {source_key!r} -> {target_key!r}: "
                f"artifact={tuple(value.shape)} model={tuple(target_state[target_key].shape)}"
            )
        target_dtype = getattr(target_state[target_key], "dtype", value.dtype)
        mapped[target_key] = value.to(dtype=target_dtype) if torch.is_floating_point(value) else value

    if unmatched:
        raise ValueError(f"Could not map {len(unmatched)} Reasoner LoRA tensors into the model; first={unmatched[0]!r}")
    if not mapped:
        raise ValueError(f"Reasoner LoRA artifact mapped zero tensors into the model: {root}")

    if target_is_distributed:
        try:
            incompatible = set_model_state_dict(
                model,
                model_state_dict=dict(mapped),
                options=StateDictOptions(strict=False, full_state_dict=True),
            )
        except Exception as exc:
            raise RuntimeError("Failed to load the initial Reasoner LoRA into the distributed model") from exc
    else:
        incompatible = model.load_state_dict(mapped, strict=False)
    unexpected = set(getattr(incompatible, "unexpected_keys", ()))
    if unexpected:
        raise ValueError(f"Unexpected Reasoner LoRA keys after load: {sorted(unexpected)[:3]}")
    log.info(f"Loaded initial Reasoner LoRA: tensors={len(mapped)} path={str(root)!r}")
    return len(mapped)


class LoraInjectedLinear(nn.Linear):
    """nn.Linear with sibling lora_A and lora_B that preserves the original ``.weight`` key.

    State-dict keys for a module at ``<path>``:
      ``<path>.weight``        — original Linear weight (unchanged key)
      ``<path>.bias``          — original Linear bias (if present)
      ``<path>.lora_A.weight`` — low-rank down-projection
      ``<path>.lora_B.weight`` — low-rank up-projection

    Forward computes ``y = base(x) + (alpha / r) * lora_B(lora_A(x))`` by
    default.  ``set_lora_runtime_enabled(False)`` temporarily selects the
    base-only projection without changing parameter ``requires_grad`` flags or
    checkpoint keys.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        # Reuse base's geometry. Inherit nn.Linear so ``super().forward(x)``
        # dispatches to the standard F.linear path.
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device="meta",
        )
        # Replace nn.Linear's freshly-allocated parameters with the base's
        # existing ones. On meta device this is a no-op for memory; on a
        # real device this preserves the pretrained weight identity.
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        # Sibling submodules. Using bias=False nn.Linear gives us a single
        # ``weight`` parameter and a clean ``.lora_A.weight`` state-dict key.
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, device="meta")
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, device="meta")
        self._lora_rank = int(rank)
        self._lora_alpha = float(alpha)
        # Runtime-only phase switch.  This is intentionally a plain Python
        # flag rather than a parameter/buffer: it must not add a checkpoint
        # key or participate in FSDP sharding.  Reasoner-WAM can temporarily
        # route a WAM forward through the frozen/base projection while the
        # Reasoner phase continues to use the adapter.  The default keeps the
        # historical ``base + LoRA`` behavior.
        self._lora_runtime_enabled = True

    @property
    def _lora_scale(self) -> float:
        return self._lora_alpha / self._lora_rank

    def forward_base(self, x: torch.Tensor) -> torch.Tensor:
        """Apply only the frozen/base linear projection."""
        return F.linear(x, self.weight, self.bias)

    @property
    def lora_runtime_enabled(self) -> bool:
        """Whether this module currently adds its LoRA delta during forward."""

        return self._lora_runtime_enabled

    def set_lora_runtime_enabled(self, enabled: bool) -> None:
        """Enable or disable the adapter contribution for subsequent forwards.

        This does not alter ``requires_grad`` on either the base or adapter
        parameters.  It is therefore safe to use around one phase of a
        two-forward step and restore immediately afterwards.
        """

        self._lora_runtime_enabled = bool(enabled)

    def forward_with_adapter_mask(self, x: torch.Tensor, adapter_mask: torch.Tensor) -> torch.Tensor:
        """Apply the LoRA delta only at positions selected by ``adapter_mask``.

        ``adapter_mask`` matches ``x.shape[:-1]``. This keeps unselected token
        rows numerically identical to the base projection while preserving the
        usual shared LoRA weights for selected rows.
        """
        if adapter_mask.shape != x.shape[:-1]:
            raise ValueError(
                f"LoRA adapter mask shape {tuple(adapter_mask.shape)} must match input prefix shape {tuple(x.shape[:-1])}."
            )
        base_out = self.forward_base(x)
        if not self._lora_runtime_enabled:
            return base_out
        lora_out = self.lora_B(self.lora_A(x))
        lora_out = lora_out * adapter_mask.to(device=x.device, dtype=lora_out.dtype).unsqueeze(-1)
        return base_out + self._lora_scale * lora_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.forward_base(x)
        if not self._lora_runtime_enabled:
            return base_out
        lora_out = self.lora_B(self.lora_A(x))
        return base_out + self._lora_scale * lora_out


def _inject_lora_inplace(
    network: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: int,
) -> int:
    """Replace each ``<target>`` ``nn.Linear`` child in-place with ``LoraInjectedLinear``.

    Match is by exact child name (e.g., ``q_proj_moe_gen``), not substring.
    Snapshots ``named_modules()`` before mutating the tree so newly-inserted
    LoRA submodules are not re-visited.
    """
    target_set = set(target_modules)
    replaced = 0
    for _parent_name, parent in list(network.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in target_set and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoraInjectedLinear(child, rank, alpha))
                replaced += 1
    return replaced


def inject_lora_pre_fsdp(
    network: torch.nn.Module,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_target_modules: str,
    freeze_non_lora: bool = True,
) -> torch.nn.Module:
    """Inject LoRA adapters into ``network`` BEFORE FSDP wrap on meta device.

    Must be called on the meta-device network (pre-FSDP) so the injector sees
    unsharded weight shapes; injecting after FSDP causes ``lora_B`` to be
    constructed with the per-rank shard size (e.g., 8192/8=1024) and triggers
    a shape mismatch at forward time.

    ``lora_A`` and ``lora_B`` parameters are left uninitialized on meta;
    the caller must initialize them AFTER
    ``to_empty(device="cuda") + init_weights(buffer_device="cuda")``
    via ``init_lora_weights_post_materialization``.

    By default, freezes every non-LoRA parameter so the optimizer's
    ``keys_to_select=["lora_"]`` filter trains adapters only. Set
    ``freeze_non_lora=False`` for hybrid full-module + LoRA recipes; this leaves
    pre-existing ``requires_grad`` flags unchanged.
    """
    assert network is not None, "Network is not initialized"

    if lora_rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {lora_rank}")
    if lora_alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")

    target_modules_list = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
    if not target_modules_list:
        raise ValueError("LoRA target_modules cannot be empty")

    model_module_names = {name.split(".")[-1] for name, _ in network.named_modules()}
    invalid_modules = [t for t in target_modules_list if t not in model_module_names]
    if invalid_modules:
        log.warning(f"LoRA target modules not found in model: {invalid_modules}")

    log.info(f"Injecting LoRA on meta device: rank={lora_rank}, alpha={lora_alpha}, targets={target_modules_list}")

    try:
        replaced = _inject_lora_inplace(network, target_modules_list, lora_rank, lora_alpha)
    except Exception as e:
        raise RuntimeError(f"Failed to inject LoRA adapters into model: {e}") from e

    if replaced == 0:
        log.warning(f"LoRA injection replaced 0 modules — check lora_target_modules={lora_target_modules!r}")

    lora_params = 0
    frozen_params = 0
    preserved_params = 0
    for name, param in network.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            lora_params += param.numel()
        elif freeze_non_lora:
            param.requires_grad_(False)
            frozen_params += param.numel()
        else:
            preserved_params += param.numel()

    log.info(
        f"LoRA injection successful: {replaced} modules wrapped, "
        f"{lora_params:,} trainable LoRA params, "
        f"{frozen_params:,} frozen base params (freeze_non_lora={freeze_non_lora}) "
        f"and {preserved_params:,} base params with preserved grad flags"
    )
    return network


def init_lora_weights_post_materialization(network: torch.nn.Module) -> None:
    """Initialize LoRA params after ``to_empty + init_weights`` materializes them.

    The custom injector leaves lora_A/lora_B as uninitialized meta-device
    parameters. After ``to_empty(device=DEVICE)``, they have allocated but
    uninitialized memory. Init in-place (``lora_A ~ kaiming_uniform_(a=sqrt(5))``,
    ``lora_B = zeros``) and cast each pair to its wrapped base weight's dtype
    so ``F.linear(x, lora_A.weight)`` sees matching dtypes whether the base
    runs in fp32, bf16, or fp16.
    """
    for module in network.modules():
        if not isinstance(module, LoraInjectedLinear):
            continue
        base_dtype = module.weight.dtype
        torch.nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
        module.lora_A.weight.data = module.lora_A.weight.data.to(base_dtype)
        torch.nn.init.zeros_(module.lora_B.weight)
        module.lora_B.weight.data = module.lora_B.weight.data.to(base_dtype)


def freeze_vlm_base_for_soft_plan_lora(language_model: nn.Module) -> None:
    """Freeze VLM/reasoner parameters while preserving gen experts and LoRA.

    The MoT generation pathway is identified by its established ``_moe_gen``
    suffix. Calling this before or after LoRA injection is supported.
    """
    for name, param in language_model.named_parameters():
        if "_moe_gen" in name or "lora_" in name:
            continue
        param.requires_grad_(False)


def iter_lora_modules(network: nn.Module) -> Iterator[LoraInjectedLinear]:
    """Yield LoRA projections below ``network`` exactly once.

    The helper deliberately relies on ``named_modules`` rather than parameter
    names so it also works after activation-checkpoint, compiled, or FSDP
    wrappers have been installed around the MoT blocks.
    """

    for module in network.modules():
        if isinstance(module, LoraInjectedLinear):
            yield module


def set_lora_runtime_enabled(network: nn.Module, enabled: bool) -> None:
    """Set adapter routing for every LoRA projection below ``network``.

    Unlike changing ``requires_grad``, this only changes which branch is used
    in the next forward.  It is therefore suitable for alternating Reasoner
    and WAM phases that share one model and one set of LoRA parameters.
    """

    for module in iter_lora_modules(network):
        module.set_lora_runtime_enabled(enabled)


@contextmanager
def lora_runtime_context(network: nn.Module, *, enabled: bool) -> Iterator[None]:
    """Temporarily set adapter use for all LoRA projections in ``network``.

    Previous per-module states are restored even when forward/backward raises,
    which is important for a failed WAM phase and for nested evaluation calls.
    No model parameters or checkpoint state are modified.
    """

    modules = list(iter_lora_modules(network))
    previous = [module.lora_runtime_enabled for module in modules]
    for module in modules:
        module.set_lora_runtime_enabled(enabled)
    try:
        yield
    finally:
        for module, old_enabled in zip(modules, previous, strict=True):
            module.set_lora_runtime_enabled(old_enabled)
