# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io

_TRAIN_DETAIL_LOSS_ALIASES = {
    "reasoner_ce_loss": "subtask_ce_loss",
}

_OBSOLETE_REASONER_WAM_LOSS_KEYS = frozenset(
    {
        "reasoner_loss",
        "wam_loss",
        "wam_flow_loss",
        "reasoner_ce_loss_weighted",
    }
)


@dataclass
class _LossRecord:
    loss: torch.Tensor | float = 0
    iter_count: int = 0
    name: str | None = None

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0

    def get_stat(self, return_valid_mask_sum: bool = False) -> Tuple[float, float]:
        if self.iter_count == 0:
            self.loss = torch.tensor([float("nan")], device="cuda")
            self.iter_count = 1
        self.loss = self.loss.mean()
        msg_str = f"{self.name}: sum_loss={self.loss.item()}/iter_count={self.iter_count}="
        avg_loss_tensor = self.loss / self.iter_count
        # Create a mask (1 if valid, 0 if NaN or Inf)
        valid_mask = torch.tensor([torch.isfinite(avg_loss_tensor).float()], device="cuda")
        msg_str += f"avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}, "

        # Replace NaN/Inf with 0 to avoid affecting sum
        avg_loss_tensor = torch.where(
            torch.isfinite(avg_loss_tensor), avg_loss_tensor, torch.tensor([0.0], device="cuda")
        )

        # Reduce across all ranks
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)  # Sum of valid losses
        dist.all_reduce(valid_mask, op=dist.ReduceOp.SUM)  # Count of valid losses
        msg_str += f" | all_reduce: avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}"
        # Compute final average, avoiding division by zero
        if valid_mask.item() > 0:
            final_avg_loss = (avg_loss_tensor / valid_mask).item()
            valid_mask_sum = valid_mask.item()
        else:
            final_avg_loss = 0.0  # Default to zero if all values were invalid
            valid_mask_sum = 0

        avg_loss = final_avg_loss
        msg_str += f" | final: avg_loss={final_avg_loss}"
        if self.name is not None:
            log.debug(msg_str, rank0_only=False)
        self.reset()
        if return_valid_mask_sum:
            return avg_loss, valid_mask_sum
        else:
            return avg_loss


class WandbCallback(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.final_loss_log = _LossRecord()
        self.final_loss_log_per_dataset = {}
        self.final_all_loss_log = {}
        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        assert self.logging_iter_multipler > 0, "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
        self.name = "wandb_loss_log" + self.wandb_extra_tag
        self.unstable_count = torch.zeros(1, device="cuda")
        # The paired Reasoner/WAM trainer places a bounded copy of the
        # assistant gold text in ``output_batch``.  Keep it separate from the
        # scalar loss accumulators so old flow-only batches remain untouched.
        self._reasoner_subgoal_texts: list[str] = []
        self._max_reasoner_subgoal_texts = 32

    @staticmethod
    def _as_text_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if isinstance(value, torch.Tensor):
            return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
        return [str(value)]

    def _accumulate_loss_details(self, output_batch: dict[str, torch.Tensor]) -> None:
        for key, value in output_batch.items():
            # Curve can be plotted only on aggregated loss, not per-instance loss.
            if key in _OBSOLETE_REASONER_WAM_LOSS_KEYS or "loss" not in key or "per_instance" in key:
                continue
            log_key = _TRAIN_DETAIL_LOSS_ALIASES.get(key, key)
            # Prefer an explicitly supplied canonical value if a future output
            # carries both it and its legacy Reasoner name.
            if log_key != key and log_key in output_batch:
                continue
            if log_key not in self.final_all_loss_log:
                self.final_all_loss_log[log_key] = _LossRecord()
            self.final_all_loss_log[log_key].loss += value.detach().float()
            self.final_all_loss_log[log_key].iter_count += 1

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if torch.isnan(loss) or torch.isinf(loss):
            log.critical(
                f"Unstable loss {loss} at iteration {iteration}",
                rank0_only=False,
            )
            self.unstable_count += 1

        # This is intentionally conditional.  Legacy mid-soft/WAM batches do
        # not carry the key and therefore incur no text handling or W&B schema
        # change.  The trainer has already copied the Reasoner strings off the
        # CUDA batch before invoking this logical-step hook.
        reasoner_texts = self._as_text_list(output_batch.get("reasoner_gold_target_text"))
        if reasoner_texts:
            self._reasoner_subgoal_texts.extend(reasoner_texts)
            self._reasoner_subgoal_texts = self._reasoner_subgoal_texts[-self._max_reasoner_subgoal_texts :]

        dataset_name = data_batch.get("dataset_name", "default")

        if dataset_name == "default" and "__url__" in data_batch:
            # try to get the name from url
            dataset_name = ["/".join(data_batch["__url__"][0].split("/")[:-1])]

        # ``JointCosmosDataLoader`` stamps one scalar name on a complete
        # heterogeneous batch, while the legacy VFM packer emits a list of
        # per-sample names.  Normalize both forms before the per-dataset loop;
        # iterating a scalar string would otherwise log one record per
        # character (for example ``"reasoner"`` -> ``r,e,a,...``).
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]
        elif isinstance(dataset_name, tuple):
            dataset_name = list(dataset_name)
        elif not isinstance(dataset_name, (list, set)):
            dataset_name = [str(dataset_name)]
        # Preserve the legacy "one record per distinct dataset in this batch"
        # behavior while also accepting the new scalar marker.  ``dict`` keeps
        # first-seen order and avoids the non-deterministic ordering of set().
        dataset_names = list(dict.fromkeys(dataset_name))

        for single_dataset_name in dataset_names:
            if single_dataset_name not in self.final_loss_log_per_dataset:
                self.final_loss_log_per_dataset[single_dataset_name] = _LossRecord()
                self.final_loss_log_per_dataset[single_dataset_name].name = single_dataset_name
            self.final_loss_log_per_dataset[single_dataset_name].loss += loss.detach().float()
            self.final_loss_log_per_dataset[single_dataset_name].iter_count += 1

        # VLM: per-sequence loss normalization using token counts when available
        if "avg_num_assistant_tokens" in output_batch:
            per_seq_loss = (
                loss
                * output_batch["avg_num_assistant_tokens"]
                * output_batch["batch_size_local"]
                / output_batch["current_num_assistant_tokens"]
            )
            per_seq_key = f"per_seq/{dataset_name}"
            if per_seq_key not in self.final_loss_log_per_dataset:
                self.final_loss_log_per_dataset[per_seq_key] = _LossRecord()
                self.final_loss_log_per_dataset[per_seq_key].name = per_seq_key
            self.final_loss_log_per_dataset[per_seq_key].loss += per_seq_loss
            self.final_loss_log_per_dataset[per_seq_key].iter_count += 1

        self.final_loss_log.loss += loss.detach().float()
        self.final_loss_log.iter_count += 1

        self._accumulate_loss_details(output_batch)

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            avg_final_loss = self.final_loss_log.get_stat()

            avg_final_all_loss = {}
            for key in self.final_all_loss_log.keys():
                avg_final_all_loss[key] = self.final_all_loss_log[key].get_stat()

            # Step 1: Gather all dataset names across ranks
            local_dataset_names = list(self.final_loss_log_per_dataset.keys())
            all_dataset_names = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_dataset_names, local_dataset_names)

            # Step 2: Create the union of all dataset names
            union_dataset_names = set()
            for names in all_dataset_names:
                union_dataset_names.update(names)
            # Step 3: For any missing dataset name, add dummy _LossRecord with NaN loss
            union_dataset_names = sorted(list(union_dataset_names))  # This is very important!
            for dataset_name in union_dataset_names:
                if dataset_name not in self.final_loss_log_per_dataset:
                    dummy = _LossRecord()
                    dummy.loss += torch.tensor([float("nan")], device="cuda")  # Will be masked out
                    dummy.iter_count += 1
                    self.final_loss_log_per_dataset[dataset_name] = dummy

            avg_final_loss_per_dataset = {}
            for dataset_name in union_dataset_names:
                avg_loss, valid_mask_sum = self.final_loss_log_per_dataset[dataset_name].get_stat(
                    return_valid_mask_sum=True
                )
                if valid_mask_sum > 0:
                    avg_final_loss_per_dataset[dataset_name] = avg_loss

            dist.all_reduce(self.unstable_count, op=dist.ReduceOp.SUM)

            # Publish the original Reasoner target text as a W&B table.
            # Gather at the same cadence as scalar logging so every rank
            # enters the collective, while rank 0 owns the actual W&B object.
            local_texts = list(self._reasoner_subgoal_texts)
            if dist.is_available() and dist.is_initialized():
                gathered_texts: list[list[str] | None] = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_texts, local_texts)
                all_texts = [text for rows in gathered_texts if rows for text in rows]
            else:
                all_texts = local_texts
            all_texts = all_texts[-self._max_reasoner_subgoal_texts :]

            if distributed.is_rank0() and wandb.run is not None:
                info = {}
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/loss": avg_final_loss,
                        f"train{self.wandb_extra_tag}/unstable_count": self.unstable_count.item(),
                        "iteration": iteration,
                    }
                )
                for key, loss in avg_final_all_loss.items():
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}_detail/{key}": loss,
                        }
                    )
                for dataset_name, loss in avg_final_loss_per_dataset.items():
                    tag = ""
                    if "per_seq" in dataset_name:
                        tag = "_per_seq"
                        dataset_name = dataset_name.replace("per_seq/", "")
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}_per_data{tag}/{dataset_name}": loss,
                        }
                    )

                if all_texts:
                    info["reasoner/gold_subgoal_text"] = wandb.Table(
                        columns=["text"],
                        data=[[text] for text in all_texts],
                    )
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.logging_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)

            # reset unstable count
            self.unstable_count.zero_()
            self._reasoner_subgoal_texts.clear()
            self.final_loss_log_per_dataset = {}
