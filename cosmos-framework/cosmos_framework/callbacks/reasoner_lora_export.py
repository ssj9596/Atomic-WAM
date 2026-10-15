# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export standalone Reasoner LoRA/projector artifacts beside each DCP checkpoint."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from typing import Any

import torch

from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback

_LORA_PARTS = ("lora_A", "lora_B")
_PROJECTOR_PREFIXES = ("merger.", "deepstack_merger_list.")
_EXPORT_FILENAMES = (
    "reasoner_lora.safetensors",
    "reasoner_lora.json",
    "reasoner_projector.safetensors",
    "reasoner_projector.json",
)


def _atomic_link_or_copy(source: str, destination: str) -> None:
    """Publish an independent path, preferring a zero-extra-space hard link."""

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(destination)}.", dir=os.path.dirname(destination))
    os.close(fd)
    os.unlink(temporary)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json_dump(payload: dict[str, Any], destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(destination)}.", dir=os.path.dirname(destination))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def canonical_reasoner_lora_key(name: str) -> str:
    """Convert a pure-HF VLM adapter name to the Omni ``net`` namespace."""

    name = name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
    if name.startswith("model.language_model."):
        name = name.removeprefix("model.")
    if name.startswith("language_model."):
        inner = name.removeprefix("language_model.")
        if inner.startswith("model."):
            return name
        return f"language_model.model.{inner}"
    raise ValueError(f"Reasoner LoRA tensor is outside the language_model tower: {name!r}")


def canonical_reasoner_projector_key(name: str) -> str:
    """Convert a pure-HF visual projector name to the Omni ``net`` namespace."""

    name = name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
    for prefix in ("model.model.visual.", "model.visual.", "visual."):
        if name.startswith(prefix):
            inner = name.removeprefix(prefix)
            if inner.startswith(_PROJECTOR_PREFIXES):
                return f"language_model.visual.{inner}"
            break
    raise ValueError(f"Reasoner projector tensor is outside visual merger modules: {name!r}")


def _validate_adapter(tensors: dict[str, torch.Tensor], *, rank: int, targets: set[str]) -> None:
    if not tensors:
        raise ValueError("Reasoner LoRA export found no adapter tensors")

    parts_by_module: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
    for key, tensor in tensors.items():
        if not key.startswith("language_model.") or "_moe_gen" in key:
            raise ValueError(f"Invalid Reasoner LoRA key: {key!r}")
        fields = key.split(".")
        if len(fields) < 4 or fields[-1] != "weight" or fields[-2] not in _LORA_PARTS:
            raise ValueError(f"Invalid Reasoner LoRA key shape: {key!r}")
        target = fields[-3]
        if target not in targets:
            raise ValueError(f"Unexpected Reasoner LoRA target {target!r} in {key!r}; expected {sorted(targets)}")
        parts_by_module[".".join(fields[:-2])][fields[-2]] = tuple(tensor.shape)

    for module, parts in parts_by_module.items():
        if set(parts) != set(_LORA_PARTS):
            raise ValueError(f"Unpaired Reasoner LoRA tensors for {module!r}: {sorted(parts)}")
        a_shape = parts["lora_A"]
        b_shape = parts["lora_B"]
        if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != rank or b_shape[1] != rank:
            raise ValueError(
                f"Reasoner LoRA rank mismatch for {module!r}: A={a_shape}, B={b_shape}, configured rank={rank}"
            )


def gather_reasoner_lora_tensors(model: Any) -> dict[str, torch.Tensor]:
    """Gather only LoRA parameters; all ranks participate in DTensor collectives."""

    network = model.model.model
    gathered: dict[str, torch.Tensor] = {}
    for name, param in sorted(network.named_parameters(), key=lambda item: item[0]):
        if ".lora_A.weight" not in name and ".lora_B.weight" not in name:
            continue
        key = canonical_reasoner_lora_key(name)
        tensor = param.full_tensor() if isinstance(param, torch.distributed.tensor.DTensor) else param
        if distributed.is_rank0():
            gathered[key] = tensor.detach().to(device="cpu").contiguous()
    return gathered


def gather_reasoner_projector_tensors(model: Any) -> dict[str, torch.Tensor]:
    """Gather trainable Qwen visual merger parameters in the Omni namespace."""

    network = model.model.model
    gathered: dict[str, torch.Tensor] = {}
    for name, param in sorted(network.named_parameters(), key=lambda item: item[0]):
        if not param.requires_grad:
            continue
        try:
            key = canonical_reasoner_projector_key(name)
        except ValueError:
            continue
        tensor = param.full_tensor() if isinstance(param, torch.distributed.tensor.DTensor) else param
        if distributed.is_rank0():
            gathered[key] = tensor.detach().to(device="cpu").contiguous()
    return gathered


class ReasonerLoraExportCallback(Callback):
    """Write LoRA and any trained projector beside each local DCP iteration."""

    def __init__(self) -> None:
        super().__init__()
        self._current_iteration = 0
        self._pending: tuple[
            int,
            dict[str, torch.Tensor],
            dict[str, str],
            dict[str, Any],
            dict[str, torch.Tensor],
            dict[str, str],
            dict[str, Any],
        ] | None = None
        self._validation_loss_sum = 0.0
        self._validation_token_count = 0
        self._best_validation_loss = math.inf

    def _publish_named_export(
        self,
        *,
        source_dir: str,
        name: str,
        iteration: int,
        validation_loss: float | None,
    ) -> None:
        if not distributed.is_rank0():
            return
        destination_dir = os.path.join(self.config.job.path_local, "export", name)
        os.makedirs(destination_dir, exist_ok=True)
        for filename in _EXPORT_FILENAMES:
            source = os.path.join(source_dir, filename)
            destination = os.path.join(destination_dir, filename)
            if os.path.isfile(source):
                _atomic_link_or_copy(source, destination)
            elif os.path.exists(destination):
                os.unlink(destination)
        manifest = {
            "iteration": int(iteration),
            "source": os.path.relpath(source_dir, self.config.job.path_local),
            "validation_loss": validation_loss,
            "validation_perplexity": None if validation_loss is None else math.exp(min(validation_loss, 20.0)),
        }
        _atomic_json_dump(manifest, os.path.join(destination_dir, "selection.json"))
        log.info(f"Published Reasoner {name} export: {destination_dir!r} iteration={iteration}")

    def on_save_checkpoint_start(self, model: Any, iteration: int = 0) -> None:
        self._current_iteration = int(iteration)

    def on_save_checkpoint(self, model: Any, state_dict: dict[str, Any]) -> None:
        del state_dict
        if self._pending is not None:
            raise RuntimeError(
                "Previous Reasoner LoRA export is still pending checkpoint success; "
                "disable async DCP for standalone Reasoner training."
            )
        config = model.config
        if not getattr(config, "lora_enabled", False):
            return
        if self.config.checkpoint.save_to_object_store.enabled:
            raise ValueError("ReasonerLoraExportCallback currently requires local checkpoint saving")

        tensors = gather_reasoner_lora_tensors(model)
        projector_tensors = gather_reasoner_projector_tensors(model)
        if not distributed.is_rank0():
            self._pending = (self._current_iteration, {}, {}, {}, {}, {}, {})
            return

        targets = {value.strip() for value in config.lora_target_modules.split(",") if value.strip()}
        _validate_adapter(tensors, rank=int(config.lora_rank), targets=targets)

        policy = config.policy
        metadata = {
            "format": "cosmos_custom_lora_v1",
            "key_style": "network",
            "base_model": str(policy.backbone.model_name),
            "base_weights": str(policy.backbone.safetensors_path or policy.backbone.pretrained_weights.backbone_path),
            "lora_rank": str(config.lora_rank),
            "lora_alpha": str(config.lora_alpha),
            "lora_target_modules": ",".join(sorted(targets)),
            "tensor_count": str(len(tensors)),
        }
        sidecar = {
            **metadata,
            "lora_rank": int(config.lora_rank),
            "lora_alpha": int(config.lora_alpha),
            "lora_target_modules": sorted(targets),
            "tensor_count": len(tensors),
            "artifact": "reasoner_lora.safetensors",
        }
        projector_metadata: dict[str, str] = {}
        projector_sidecar: dict[str, Any] = {}
        if projector_tensors:
            projector_metadata = {
                "format": "cosmos_reasoner_projector_v1",
                "key_style": "network",
                "base_model": str(policy.backbone.model_name),
                "base_weights": str(
                    policy.backbone.safetensors_path or policy.backbone.pretrained_weights.backbone_path
                ),
                "components": "visual.merger,visual.deepstack_merger_list",
                "tensor_count": str(len(projector_tensors)),
            }
            projector_sidecar = {
                **projector_metadata,
                "tensor_count": len(projector_tensors),
                "artifact": "reasoner_projector.safetensors",
            }
        self._pending = (
            self._current_iteration,
            tensors,
            metadata,
            sidecar,
            projector_tensors,
            projector_metadata,
            projector_sidecar,
        )

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        del elapsed_time
        if self._pending is None:
            return
        if not distributed.is_rank0():
            self._pending = None
            return
        (
            pending_iteration,
            tensors,
            metadata,
            sidecar,
            projector_tensors,
            projector_metadata,
            projector_sidecar,
        ) = self._pending
        if pending_iteration != int(iteration):
            raise RuntimeError(
                f"Reasoner LoRA export iteration mismatch: pending={pending_iteration}, saved={iteration}"
            )

        iteration_dir = os.path.join(
            self.config.job.path_local,
            "checkpoints",
            f"iter_{pending_iteration:09d}",
        )
        os.makedirs(iteration_dir, exist_ok=True)
        artifact_path = os.path.join(iteration_dir, "reasoner_lora.safetensors")
        metadata_path = os.path.join(iteration_dir, "reasoner_lora.json")

        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - training dependency
            raise RuntimeError("safetensors is required for Reasoner LoRA export") from exc

        fd, temp_name = tempfile.mkstemp(prefix=".reasoner_lora.", suffix=".safetensors", dir=iteration_dir)
        os.close(fd)
        os.unlink(temp_name)
        try:
            save_file(tensors, temp_name, metadata=metadata)
            os.replace(temp_name, artifact_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

        fd, temp_name = tempfile.mkstemp(prefix=".reasoner_lora.", suffix=".json", dir=iteration_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(sidecar, stream, indent=2, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, metadata_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

        log.info(f"Reasoner LoRA export complete: tensors={len(tensors)} path={artifact_path!r}")
        if projector_tensors:
            projector_path = os.path.join(iteration_dir, "reasoner_projector.safetensors")
            projector_metadata_path = os.path.join(iteration_dir, "reasoner_projector.json")

            fd, temp_name = tempfile.mkstemp(prefix=".reasoner_projector.", suffix=".safetensors", dir=iteration_dir)
            os.close(fd)
            os.unlink(temp_name)
            try:
                save_file(projector_tensors, temp_name, metadata=projector_metadata)
                os.replace(temp_name, projector_path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)

            fd, temp_name = tempfile.mkstemp(prefix=".reasoner_projector.", suffix=".json", dir=iteration_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(projector_sidecar, stream, indent=2, ensure_ascii=False, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, projector_metadata_path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            log.info(
                f"Reasoner projector export complete: tensors={len(projector_tensors)} path={projector_path!r}"
            )
        self._publish_named_export(
            source_dir=iteration_dir,
            name="last",
            iteration=pending_iteration,
            validation_loss=None,
        )
        self._pending = None

    def on_validation_start(self, model: Any, dataloader_val: Any, iteration: int = 0) -> None:
        del model, dataloader_val, iteration
        self._validation_loss_sum = 0.0
        self._validation_token_count = 0

    def on_validation_step_end(
        self,
        model: Any,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, iteration
        labels = output_batch.get("labels")
        if not isinstance(labels, torch.Tensor):
            raise RuntimeError("Reasoner best-export validation requires output_batch['labels']")
        local_tokens = torch.count_nonzero(labels[:, 1:] != -100).to(device=loss.device, dtype=torch.long)
        loss_sum = loss.detach().float().clone()
        token_sum = local_tokens.clone()
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(token_sum, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
        batch_loss = float(loss_sum.item()) / float(world_size)
        batch_tokens = int(token_sum.item())
        self._validation_loss_sum += batch_loss * batch_tokens
        self._validation_token_count += batch_tokens

    def on_validation_end(self, model: Any, iteration: int = 0) -> None:
        del model
        if self._validation_token_count <= 0:
            raise RuntimeError("Reasoner validation produced zero supervised assistant tokens")
        validation_loss = self._validation_loss_sum / self._validation_token_count
        perplexity = math.exp(min(validation_loss, 20.0))
        log.info(
            f"Reasoner validation token CE={validation_loss:.6f} perplexity={perplexity:.6f} "
            f"tokens={self._validation_token_count} iteration={iteration}"
        )
        if distributed.is_rank0():
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "val/reasoner_token_ce": validation_loss,
                            "val/reasoner_perplexity": perplexity,
                            "val/reasoner_tokens": self._validation_token_count,
                        },
                        step=iteration,
                    )
            except ImportError:
                pass
        if math.isinf(self._best_validation_loss):
            best_manifest = os.path.join(self.config.job.path_local, "export", "best", "selection.json")
            if os.path.isfile(best_manifest):
                try:
                    with open(best_manifest, encoding="utf-8") as stream:
                        previous = json.load(stream).get("validation_loss")
                    if isinstance(previous, (float, int)) and math.isfinite(float(previous)):
                        self._best_validation_loss = float(previous)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    log.warning(f"Ignoring unreadable Reasoner best-export manifest {best_manifest!r}: {exc}")
        if validation_loss >= self._best_validation_loss:
            return
        source_dir = os.path.join(self.config.job.path_local, "checkpoints", f"iter_{int(iteration):09d}")
        if not os.path.isfile(os.path.join(source_dir, "reasoner_lora.safetensors")):
            raise RuntimeError(
                "Best Reasoner export requires validation_iter to coincide with a successful checkpoint save; "
                f"missing adapter export under {source_dir!r}."
            )
        self._publish_named_export(
            source_dir=source_dir,
            name="best",
            iteration=int(iteration),
            validation_loss=validation_loss,
        )
        self._best_validation_loss = validation_loss


__all__ = [
    "ReasonerLoraExportCallback",
    "canonical_reasoner_lora_key",
    "canonical_reasoner_projector_key",
    "gather_reasoner_lora_tensors",
    "gather_reasoner_projector_tensors",
]
