# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CUDA image augmentation for action-policy training batches."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as v2_f

from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback


@dataclass(frozen=True)
class MosaicCropParams:
    """One shared crop for the wrist/top and exterior/bottom mosaic views."""

    height: int
    width: int
    top_height: int
    bottom_height: int
    left_width: int
    right_width: int
    wrist_box: tuple[int, int, int, int]
    exterior_box: tuple[int, int, int, int]


class GPUImageAugmentationCallback(Callback):
    """Apply shared view-aware crop/resize and ColorJitter after H2D.

    Action videos arrive as uint8 tensors with shape ``[C, T, H, W]`` nested
    under ``data["video"]``. One crop offset and one torchvision v2 ColorJitter
    parameter set are sampled per local batch. The top wrist view and the two
    bottom exterior views are cropped independently with the same normalized
    offset, resized back into their original slots, and recomposed without
    moving camera boundaries. All frames, views, and samples in the local batch
    therefore share the same augmentation parameters.
    """

    _CAMERA_TO_BASE_KEY = "exterior_camera_to_base"
    _CAMERA_INTRINSICS_KEY = "exterior_camera_intrinsics"
    _CAMERA_SOURCE_SIZE_KEY = "exterior_camera_source_size"

    def __init__(
        self,
        enabled: bool = False,
        chunk_size: int = 4,
        brightness: float = 0.3,
        contrast: float = 0.4,
        saturation: float = 0.5,
        hue: float = 0.08,
        crop_scale: float = 0.95,
        camera_dropout_rate: float = 0.0,
        log_timing: bool = True,
        rebuild_vit_pixel_values: bool = False,
        vit_patch_size: int = 16,
        vit_temporal_patch_size: int = 2,
        vit_merge_size: int = 2,
        vit_image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        vit_image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()
        if int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.enabled = bool(enabled)
        self.chunk_size = int(chunk_size)
        self.crop_scale = float(crop_scale)
        if not 0.0 < self.crop_scale <= 1.0:
            raise ValueError(f"crop_scale must be in (0, 1], got {crop_scale}")
        self.camera_dropout_rate = float(camera_dropout_rate)
        if not 0.0 <= self.camera_dropout_rate <= 1.0:
            raise ValueError(f"camera_dropout_rate must be in [0, 1], got {camera_dropout_rate}")
        self.log_timing = bool(log_timing)
        self.rebuild_vit_pixel_values = bool(rebuild_vit_pixel_values)
        self.vit_patch_size = int(vit_patch_size)
        self.vit_temporal_patch_size = int(vit_temporal_patch_size)
        self.vit_merge_size = int(vit_merge_size)
        if self.vit_patch_size < 1 or self.vit_temporal_patch_size < 1 or self.vit_merge_size < 1:
            raise ValueError("ViT patch, temporal-patch, and merge sizes must all be positive")
        if len(vit_image_mean) != 3 or len(vit_image_std) != 3:
            raise ValueError("ViT image mean/std must each contain exactly three RGB values")
        self.vit_image_mean = tuple(float(value) for value in vit_image_mean)
        self.vit_image_std = tuple(float(value) for value in vit_image_std)
        self._color_jitter = v2.ColorJitter(
            brightness=float(brightness),
            contrast=float(contrast),
            saturation=float(saturation),
            hue=float(hue),
        )
        self._pending_events: deque[tuple[torch.cuda.Event, torch.cuda.Event]] = deque()
        self._elapsed_ms = 0.0
        self._timed_batches = 0
        self._logged_startup = False
        self._vit_grid_cache: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
        self._vit_normalization_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _collect_tensor_leaves(value: Any, *, key: str) -> list[torch.Tensor]:
        """Return tensor leaves from the nested containers emitted by VFMListCollator."""
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, (list, tuple)):
            tensors: list[torch.Tensor] = []
            for item in value:
                tensors.extend(GPUImageAugmentationCallback._collect_tensor_leaves(item, key=key))
            return tensors
        if value is None:
            return []
        raise TypeError(f"Expected data[{key!r}] to contain tensors/lists/tuples, got {type(value).__name__}")

    @staticmethod
    def _replace_tensor_leaves(value: Any, replacements: list[torch.Tensor], index: list[int]) -> Any:
        """Replace nested tensor leaves in traversal order while retaining container structure."""
        if isinstance(value, torch.Tensor):
            if index[0] >= len(replacements):
                raise ValueError("Not enough replacement tensors for nested ViT pixel_values")
            replacement = replacements[index[0]]
            index[0] += 1
            return replacement
        if isinstance(value, list):
            return [GPUImageAugmentationCallback._replace_tensor_leaves(item, replacements, index) for item in value]
        if isinstance(value, tuple):
            return tuple(
                GPUImageAugmentationCallback._replace_tensor_leaves(item, replacements, index) for item in value
            )
        if value is None:
            return None
        raise TypeError(f"Expected nested tensor/list/tuple structure, got {type(value).__name__}")

    def _get_vit_grid(
        self,
        video: torch.Tensor,
        image_grid_thw: torch.Tensor,
        content_size: tuple[int, int],
    ) -> tuple[int, int, int]:
        """Read and cache the fixed Qwen image grid for one source image size."""
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f"Expected RGB action video [3,T,H,W], got {tuple(video.shape)}")
        if video.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 action video for ViT rebuild, got {video.dtype}")
        if image_grid_thw.numel() != 3:
            raise ValueError(f"Expected one image_grid_thw triplet, got {tuple(image_grid_thw.shape)}")

        # image_grid_thw is tiny and constant for this fixed-resolution recipe.
        # Cache the first metadata read so later steps never introduce a
        # device-to-host synchronization; all pixel work remains on CUDA.
        cache_key = (int(video.shape[-2]), int(video.shape[-1]), *content_size)
        grid = self._vit_grid_cache.get(cache_key)
        if grid is None:
            grid = tuple(int(value) for value in image_grid_thw.detach().reshape(-1).cpu().tolist())
            self._vit_grid_cache[cache_key] = grid
        grid_t, grid_h, grid_w = grid
        if grid_t != 1:
            raise ValueError(f"First-frame ViT input requires grid_t=1, got {grid_t}")
        if grid_h % self.vit_merge_size != 0 or grid_w % self.vit_merge_size != 0:
            raise ValueError(f"ViT grid {(grid_h, grid_w)} must be divisible by merge_size={self.vit_merge_size}")
        return grid

    @staticmethod
    def _vit_content_size(video: torch.Tensor, image_size: torch.Tensor | None) -> tuple[int, int]:
        """Return the unpadded first-frame size used by the worker processor."""
        video_height, video_width = int(video.shape[-2]), int(video.shape[-1])
        if image_size is None:
            return video_height, video_width
        if image_size.numel() != 4:
            raise ValueError(f"Expected one image_size quadruplet, got {tuple(image_size.shape)}")
        target_height, target_width, content_height, content_width = (
            int(value) for value in image_size.detach().reshape(-1).cpu().tolist()
        )
        if (target_height, target_width) != (video_height, video_width):
            raise ValueError(
                "ViT image_size target does not match video shape: "
                f"metadata={(target_height, target_width)}, video={(video_height, video_width)}"
            )
        if not (0 < content_height <= video_height and 0 < content_width <= video_width):
            raise ValueError(
                f"Invalid ViT content size {(content_height, content_width)} for video {(video_height, video_width)}"
            )
        return content_height, content_width

    def _vit_pixel_values_shape(self, grid: tuple[int, int, int]) -> tuple[int, int]:
        grid_t, grid_h, grid_w = grid
        return (
            grid_t * grid_h * grid_w,
            3 * self.vit_temporal_patch_size * self.vit_patch_size * self.vit_patch_size,
        )

    def _patchify_vit_first_frames(
        self,
        first_frames: torch.Tensor,
        *,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Apply Qwen's resize/normalize/patchify to a same-shaped frame batch."""
        grid_t, grid_h, grid_w = grid
        target_h = grid_h * self.vit_patch_size
        target_w = grid_w * self.vit_patch_size
        if first_frames.shape[-2:] != (target_h, target_w):
            first_frames = v2_f.resize(
                first_frames,
                [target_h, target_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        normalization = self._vit_normalization_cache.get(first_frames.device)
        if normalization is None:
            mean = torch.tensor(self.vit_image_mean, dtype=torch.float32, device=first_frames.device) * 255.0
            std = torch.tensor(self.vit_image_std, dtype=torch.float32, device=first_frames.device) * 255.0
            normalization = (mean, std)
            self._vit_normalization_cache[first_frames.device] = normalization
        mean, std = normalization
        patches = (first_frames.to(dtype=torch.float32) - mean[None, :, None, None]) / std[None, :, None, None]
        patches = patches.unsqueeze(1)
        if self.vit_temporal_patch_size > 1:
            patches = patches.repeat(1, self.vit_temporal_patch_size, 1, 1, 1)

        batch_size = int(patches.shape[0])
        channel = int(patches.shape[2])
        patches = patches.view(
            batch_size,
            grid_t,
            self.vit_temporal_patch_size,
            channel,
            grid_h // self.vit_merge_size,
            self.vit_merge_size,
            self.vit_patch_size,
            grid_w // self.vit_merge_size,
            self.vit_merge_size,
            self.vit_patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * self.vit_temporal_patch_size * self.vit_patch_size * self.vit_patch_size,
        )

    def _vit_first_frame_to_pixel_values(
        self,
        video: torch.Tensor,
        image_grid_thw: torch.Tensor,
        existing_pixel_values: torch.Tensor | None = None,
        image_size: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rebuild Qwen3-VL image patches from the augmented first frame.

        This is the CUDA equivalent of ``Qwen2VLImageProcessorFast`` for the
        Nano processor configuration.  It intentionally uses torchvision's
        bicubic antialiased resize, the same fused normalization, and the same
        view/permute order as the Hugging Face worker-side processor.
        """
        content_height, content_width = self._vit_content_size(video, image_size)
        grid = self._get_vit_grid(video, image_grid_thw, (content_height, content_width))
        target_size = (grid[1] * self.vit_patch_size, grid[2] * self.vit_patch_size)
        if (
            existing_pixel_values is not None
            and existing_pixel_values.numel() == 0
            and (content_height, content_width) != target_size
        ):
            raise ValueError(
                "Bit-exact deferred GPU ViT generation requires an already smart-resized first frame; "
                f"content={(content_height, content_width)}, target={target_size}"
            )
        first_frame = video[:, 0, :content_height, :content_width].unsqueeze(0)
        flattened = self._patchify_vit_first_frames(first_frame, grid=grid)[0]

        target_dtype = torch.float32
        if existing_pixel_values is not None and existing_pixel_values.numel() > 0:
            expected_shape = tuple(existing_pixel_values.shape)
            if len(expected_shape) == 3 and expected_shape[0] == 1:
                flattened = flattened.unsqueeze(0)
            if tuple(flattened.shape) != expected_shape:
                raise ValueError(
                    f"GPU-rebuilt ViT pixel_values shape {tuple(flattened.shape)} does not match "
                    f"worker output {expected_shape}"
                )
            if existing_pixel_values.is_floating_point():
                target_dtype = existing_pixel_values.dtype
        elif tuple(flattened.shape) != self._vit_pixel_values_shape(grid):
            raise AssertionError(
                f"GPU-generated ViT pixel_values shape {tuple(flattened.shape)} does not match grid {grid}"
            )
        return flattened.to(dtype=target_dtype)

    def _vit_video_list_to_pixel_values(
        self,
        videos: list[torch.Tensor],
        image_grids: list[torch.Tensor],
        existing_pixel_values: list[torch.Tensor],
        image_sizes: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Batched/chunked ViT rebuild to avoid one CUDA kernel sequence per sample."""
        if image_sizes is None:
            image_sizes = [None] * len(videos)
        grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        grids: list[tuple[int, int, int]] = []
        content_sizes: list[tuple[int, int]] = []
        expected_shapes: list[tuple[int, ...]] = []
        for index, (video, image_grid, pixels, image_size) in enumerate(
            zip(videos, image_grids, existing_pixel_values, image_sizes, strict=True)
        ):
            content_size = self._vit_content_size(video, image_size)
            grid = self._get_vit_grid(video, image_grid, content_size)
            target_size = (grid[1] * self.vit_patch_size, grid[2] * self.vit_patch_size)
            if pixels.numel() == 0 and content_size != target_size:
                raise ValueError(
                    "Bit-exact deferred GPU ViT generation requires an already smart-resized first frame; "
                    f"content={content_size}, target={target_size}"
                )
            grids.append(grid)
            content_sizes.append(content_size)
            target_dtype = pixels.dtype if pixels.numel() > 0 and pixels.is_floating_point() else torch.float32
            if pixels.numel() > 0:
                expected_shape = tuple(pixels.shape)
            else:
                expected_shape = self._vit_pixel_values_shape(grid)
            expected_shapes.append(expected_shape)
            grouped[
                (
                    video.device,
                    video.dtype,
                    content_size,
                    grid,
                    target_dtype,
                    expected_shape,
                )
            ].append(index)

        rebuilt: list[torch.Tensor | None] = [None] * len(videos)
        for indexes in grouped.values():
            grid = grids[indexes[0]]
            for offset in range(0, len(indexes), self.chunk_size):
                chunk_indexes = indexes[offset : offset + self.chunk_size]
                content_height, content_width = content_sizes[chunk_indexes[0]]
                first_frames = torch.stack(
                    [videos[index][:, 0, :content_height, :content_width] for index in chunk_indexes], dim=0
                )
                flattened_batch = self._patchify_vit_first_frames(first_frames, grid=grid)
                for chunk_index, original_index in enumerate(chunk_indexes):
                    flattened = flattened_batch[chunk_index]
                    existing = existing_pixel_values[original_index]
                    if existing.numel() > 0 and existing.ndim == 3 and existing.shape[0] == 1:
                        flattened = flattened.unsqueeze(0)
                    expected_shape = expected_shapes[original_index]
                    if tuple(flattened.shape) != expected_shape:
                        raise ValueError(
                            f"GPU-rebuilt ViT pixel_values shape {tuple(flattened.shape)} does not match "
                            f"expected shape {expected_shape}"
                        )
                    target_dtype = (
                        existing.dtype if existing.numel() > 0 and existing.is_floating_point() else torch.float32
                    )
                    rebuilt[original_index] = flattened.to(dtype=target_dtype)

        if any(item is None for item in rebuilt):
            raise AssertionError("Failed to rebuild every ViT pixel_values tensor")
        return [item for item in rebuilt if item is not None]

    def _vit_context_list_to_pixel_values(
        self,
        contexts: list[torch.Tensor],
        image_grids: list[torch.Tensor],
        existing_pixel_values: list[torch.Tensor],
        image_sizes: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Rebuild flattened Qwen patches for one or more images per sample."""
        if image_sizes is None:
            image_sizes = [None] * len(contexts)
        flat_videos: list[torch.Tensor] = []
        flat_grids: list[torch.Tensor] = []
        flat_pixels: list[torch.Tensor] = []
        flat_sizes: list[torch.Tensor | None] = []
        regroup: list[list[int]] = []
        for context, grid, pixels, image_size in zip(
            contexts, image_grids, existing_pixel_values, image_sizes, strict=True
        ):
            if context.ndim != 4 or context.shape[1] != 3:
                raise ValueError(f"Expected ViT context [N,3,H,W], got {tuple(context.shape)}")
            grid_rows = grid.reshape(-1, 3)
            if grid_rows.shape[0] != context.shape[0]:
                raise ValueError(
                    "ViT context/image_grid_thw count mismatch: "
                    f"images={context.shape[0]}, grids={grid_rows.shape[0]}"
                )
            patch_counts = [int(row.prod().item()) for row in grid_rows]
            if pixels.numel() == 0:
                pixel_slices = [pixels] * len(patch_counts)
            else:
                if pixels.ndim == 3 and pixels.shape[0] == 1:
                    pixels = pixels[0]
                if pixels.ndim != 2 or pixels.shape[0] != sum(patch_counts):
                    raise ValueError(
                        "Expected flattened multi-image pixel_values with one row per patch; "
                        f"got shape={tuple(pixels.shape)}, expected_rows={sum(patch_counts)}"
                    )
                start = 0
                pixel_slices = []
                for count in patch_counts:
                    pixel_slices.append(pixels[start : start + count])
                    start += count
            indexes: list[int] = []
            for frame, grid_row, pixel_slice in zip(context, grid_rows, pixel_slices, strict=True):
                indexes.append(len(flat_videos))
                flat_videos.append(frame.unsqueeze(1))
                flat_grids.append(grid_row)
                flat_pixels.append(pixel_slice)
                flat_sizes.append(image_size)
            regroup.append(indexes)

        rebuilt_flat = self._vit_video_list_to_pixel_values(
            flat_videos,
            flat_grids,
            flat_pixels,
            flat_sizes,
        )
        rebuilt: list[torch.Tensor] = []
        for indexes, original in zip(regroup, existing_pixel_values, strict=True):
            merged = torch.cat([rebuilt_flat[index] for index in indexes], dim=0)
            if original.numel() > 0 and original.ndim == 3 and original.shape[0] == 1:
                merged = merged.unsqueeze(0)
            rebuilt.append(merged)
        return rebuilt

    def _rebuild_vit_pixel_values(self, data: dict[str, Any], videos: list[torch.Tensor]) -> None:
        """Create or synchronize first-frame ViT inputs from the training videos."""
        pixel_values = self._collect_tensor_leaves(data["pixel_values"], key="pixel_values")
        image_grids = self._collect_tensor_leaves(data.get("image_grid_thw"), key="image_grid_thw")
        if len(videos) != len(pixel_values) or len(videos) != len(image_grids):
            raise ValueError(
                "GPU ViT augmentation requires one video, pixel_values tensor, and image_grid_thw tensor per sample; "
                f"got videos={len(videos)}, pixel_values={len(pixel_values)}, image_grid_thw={len(image_grids)}"
            )

        image_sizes: list[torch.Tensor | None]
        if "image_size" in data:
            collected_sizes = self._collect_tensor_leaves(data["image_size"], key="image_size")
            if len(collected_sizes) != len(videos):
                raise ValueError(
                    f"GPU ViT generation requires one image_size per video, got {len(collected_sizes)} for {len(videos)}"
                )
            image_sizes = list(collected_sizes)
        else:
            image_sizes = [None] * len(videos)

        contexts = data.get("vit_images")
        if contexts is None:
            rebuilt = self._vit_video_list_to_pixel_values(videos, image_grids, pixel_values, image_sizes)
        else:
            context_leaves = self._collect_tensor_leaves(contexts, key="vit_images")
            if len(context_leaves) != len(videos):
                raise ValueError(
                    "GPU ViT generation requires one vit_images tensor per video, "
                    f"got {len(context_leaves)} for {len(videos)} videos"
                )
            rebuilt = self._vit_context_list_to_pixel_values(
                context_leaves,
                image_grids,
                pixel_values,
                image_sizes,
            )
        replacement_index = [0]
        data["pixel_values"] = self._replace_tensor_leaves(data["pixel_values"], rebuilt, replacement_index)
        if replacement_index[0] != len(rebuilt):
            raise AssertionError("Did not consume every rebuilt ViT pixel_values tensor")

    @staticmethod
    def _flatten_video_tree(value: Any, tensors: list[torch.Tensor]) -> tuple[str, Any]:
        """Flatten nested action-loader video containers and retain a rebuild spec."""
        if isinstance(value, torch.Tensor):
            if value.ndim == 4:
                index = len(tensors)
                tensors.append(value)
                return ("tensor", index)
            if value.ndim == 5:
                indexes = []
                for item in value:
                    indexes.append(len(tensors))
                    tensors.append(item)
                return ("batch", indexes)
            raise ValueError(f"Expected video tensor shape [C,T,H,W] or [B,C,T,H,W], got {tuple(value.shape)}")
        if isinstance(value, list):
            return ("list", [GPUImageAugmentationCallback._flatten_video_tree(item, tensors) for item in value])
        if isinstance(value, tuple):
            return ("tuple", [GPUImageAugmentationCallback._flatten_video_tree(item, tensors) for item in value])
        if value is None:
            return ("none", None)
        raise TypeError(f"Expected data['video'] to contain tensors, lists, or tuples, got {type(value).__name__}")

    @staticmethod
    def _rebuild_video_tree(spec: tuple[str, Any], tensors: list[torch.Tensor]) -> Any:
        kind, payload = spec
        if kind == "tensor":
            return tensors[payload]
        if kind == "batch":
            return torch.stack([tensors[index] for index in payload], dim=0)
        if kind == "list":
            return [GPUImageAugmentationCallback._rebuild_video_tree(item, tensors) for item in payload]
        if kind == "tuple":
            return tuple(GPUImageAugmentationCallback._rebuild_video_tree(item, tensors) for item in payload)
        if kind == "none":
            return None
        raise AssertionError(f"Unknown video rebuild spec: {kind}")

    @staticmethod
    def _sample_shared_offsets() -> tuple[float, float]:
        """Sample one normalized vertical/horizontal crop offset for a local batch."""
        return float(torch.rand(()).item()), float(torch.rand(()).item())

    def _crop_box(self, height: int, width: int, offset_y: float, offset_x: float) -> tuple[int, int, int, int]:
        crop_height = max(1, int(height * self.crop_scale))
        crop_width = max(1, int(width * self.crop_scale))
        max_top = height - crop_height
        max_left = width - crop_width
        top = min(max_top, int(offset_y * (max_top + 1))) if max_top > 0 else 0
        left = min(max_left, int(offset_x * (max_left + 1))) if max_left > 0 else 0
        return top, left, crop_height, crop_width

    def _make_mosaic_crop_params(self, video: torch.Tensor) -> MosaicCropParams:
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f"Expected RGB action video [3,T,H,W], got {tuple(video.shape)}")
        height, width = (int(video.shape[-2]), int(video.shape[-1]))
        if height < 3 or width < 2:
            raise ValueError(f"Action mosaic must be at least 3x2 pixels, got {height}x{width}")
        bottom_height = height // 3
        top_height = height - bottom_height
        left_width = width // 2
        right_width = width - left_width
        if left_width != right_width:
            raise ValueError(f"CUDA action augmentation requires an even mosaic width, got {width}")
        offset_y, offset_x = self._sample_shared_offsets()
        return MosaicCropParams(
            height=height,
            width=width,
            top_height=top_height,
            bottom_height=bottom_height,
            left_width=left_width,
            right_width=right_width,
            wrist_box=self._crop_box(top_height, width, offset_y, offset_x),
            exterior_box=self._crop_box(bottom_height, left_width, offset_y, offset_x),
        )

    @staticmethod
    def _crop_resize_view(
        value: torch.Tensor,
        box: tuple[int, int, int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        top, left, height, width = box
        value = v2_f.crop(value, top=top, left=left, height=height, width=width)
        if value.shape[-2:] != output_size:
            value = v2_f.resize(
                value,
                list(output_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        return value

    def _augment_video_chunk(
        self,
        videos: list[torch.Tensor],
        crop: MosaicCropParams,
        color_params: dict[str, Any],
    ) -> torch.Tensor:
        # [B,C,T,H,W] -> [B,T,C,H,W], the layout expected by torchvision v2.
        batch = torch.stack(videos, dim=0).permute(0, 2, 1, 3, 4)
        wrist = batch[..., : crop.top_height, :]
        left = batch[..., crop.top_height :, : crop.left_width]
        right = batch[..., crop.top_height :, crop.left_width :]
        wrist = self._crop_resize_view(wrist, crop.wrist_box, (crop.top_height, crop.width))
        left = self._crop_resize_view(left, crop.exterior_box, (crop.bottom_height, crop.left_width))
        right = self._crop_resize_view(right, crop.exterior_box, (crop.bottom_height, crop.right_width))
        mosaic = torch.cat((wrist, torch.cat((left, right), dim=-1)), dim=-2)
        # ColorJitter.make_params() is called once outside the chunk loop. Reusing
        # its scalar factors and function order keeps the entire local batch in sync.
        mosaic = self._color_jitter.transform(mosaic, color_params)
        return mosaic.permute(0, 2, 1, 3, 4)

    def _augment_tensor_list_and_params(
        self,
        tensors: list[torch.Tensor],
        *,
        require_cuda: bool,
    ) -> tuple[list[torch.Tensor], MosaicCropParams, dict[str, Any]]:
        """Augment one same-shaped local batch in chunks with shared parameters."""
        if not tensors:
            raise ValueError("Expected at least one action video to augment")
        reference = tensors[0]
        for tensor in tensors:
            if tensor.dtype != torch.uint8:
                raise TypeError(f"GPU image augmentation requires uint8 video input, got {tensor.dtype}")
            if require_cuda and not tensor.is_cuda:
                raise RuntimeError(f"GPU image augmentation requires CUDA video input, got device={tensor.device}")
            if tensor.device != reference.device or tensor.shape != reference.shape:
                raise ValueError("All action videos in one augmented local batch must have the same device and shape")

        crop = self._make_mosaic_crop_params(reference)
        color_params = self._color_jitter.make_params([reference.permute(1, 0, 2, 3)])
        augmented: list[torch.Tensor] = []
        with torch.no_grad():
            for offset in range(0, len(tensors), self.chunk_size):
                chunk = tensors[offset : offset + self.chunk_size]
                augmented.extend(self._augment_video_chunk(chunk, crop, color_params).unbind(0))
        return augmented, crop, color_params

    def _augment_tensor_list(self, tensors: list[torch.Tensor], *, require_cuda: bool) -> list[torch.Tensor]:
        augmented, _, _ = self._augment_tensor_list_and_params(tensors, require_cuda=require_cuda)
        return augmented

    def _augment_vit_contexts(
        self,
        contexts: list[torch.Tensor],
        crop: MosaicCropParams,
        color_params: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Apply action-video augmentation to each sample's ViT image history."""
        augmented: list[torch.Tensor] = []
        for context in contexts:
            if context.ndim != 4 or context.shape[1] != 3:
                raise ValueError(f"Expected ViT context [N,3,H,W], got {tuple(context.shape)}")
            pseudo_video = context.permute(1, 0, 2, 3).contiguous()
            updated = self._augment_video_chunk([pseudo_video], crop, color_params)[0]
            augmented.append(updated.permute(1, 0, 2, 3).contiguous())
        return augmented

    def augment_video_tree(self, value: Any, *, require_cuda: bool = True) -> Any:
        """Augment a loader-style nested video value without changing its structure."""
        tensors: list[torch.Tensor] = []
        spec = self._flatten_video_tree(value, tensors)
        augmented = self._augment_tensor_list(tensors, require_cuda=require_cuda)
        return self._rebuild_video_tree(spec, augmented)

    @classmethod
    def _metadata_items(
        cls,
        data: dict[str, Any],
        key: str,
        item_shape: tuple[int, ...],
    ) -> list[torch.Tensor]:
        leaves = cls._collect_tensor_leaves(data.get(key), key=key)
        items: list[torch.Tensor] = []
        for leaf in leaves:
            if tuple(leaf.shape) == item_shape:
                items.append(leaf)
            elif leaf.ndim == len(item_shape) + 1 and tuple(leaf.shape[1:]) == item_shape:
                items.extend(leaf.unbind(0))
            else:
                raise ValueError(f"Expected {key} item shape {item_shape}, got {tuple(leaf.shape)}")
        return items

    @staticmethod
    def _update_intrinsics_for_crop_resize(
        intrinsics: torch.Tensor,
        source_sizes: torch.Tensor,
        *,
        target_height: int,
        target_width: int,
        crop_box: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Scale source K to a view, then apply crop and resize back to that view."""
        if intrinsics.ndim != 4 or intrinsics.shape[-3:] != (2, 3, 3):
            raise ValueError(f"Expected intrinsics [B,2,3,3], got {tuple(intrinsics.shape)}")
        if source_sizes.shape != (*intrinsics.shape[:2], 2):
            raise ValueError(f"Expected source_sizes {(*intrinsics.shape[:2], 2)}, got {tuple(source_sizes.shape)}")
        top, left, crop_height, crop_width = crop_box
        source_sizes = source_sizes.to(device=intrinsics.device, dtype=intrinsics.dtype)
        source_height = source_sizes[..., 0]
        source_width = source_sizes[..., 1]
        if bool((source_height <= 0).any() or (source_width <= 0).any()):
            raise ValueError("Camera source sizes must be positive")

        updated = intrinsics.clone()
        view_scale_x = float(target_width) / source_width
        view_scale_y = float(target_height) / source_height
        updated[..., 0, 0] *= view_scale_x
        updated[..., 0, 2] *= view_scale_x
        updated[..., 1, 1] *= view_scale_y
        updated[..., 1, 2] *= view_scale_y

        crop_scale_x = float(target_width) / float(crop_width)
        crop_scale_y = float(target_height) / float(crop_height)
        updated[..., 0, 0] *= crop_scale_x
        updated[..., 0, 2] = (updated[..., 0, 2] - float(left)) * crop_scale_x
        updated[..., 1, 1] *= crop_scale_y
        updated[..., 1, 2] = (updated[..., 1, 2] - float(top)) * crop_scale_y
        return updated

    @staticmethod
    def _build_batched_plucker_raymaps(
        camera_to_base: torch.Tensor,
        intrinsics: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Vectorized equivalent of build_plucker_raymap for [B,2] cameras."""
        if camera_to_base.ndim != 4 or camera_to_base.shape[-3:] != (2, 4, 4):
            raise ValueError(f"Expected camera_to_base [B,2,4,4], got {tuple(camera_to_base.shape)}")
        if intrinsics.shape != (*camera_to_base.shape[:2], 3, 3):
            raise ValueError(f"Expected intrinsics {(*camera_to_base.shape[:2], 3, 3)}, got {tuple(intrinsics.shape)}")
        dtype = intrinsics.dtype
        device = intrinsics.device
        y_coords = torch.arange(height, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(width, device=device, dtype=dtype) + 0.5
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid_x = grid_x[None, None]
        grid_y = grid_y[None, None]
        directions_camera = torch.stack(
            (
                (grid_x - intrinsics[..., 0, 2, None, None]) / intrinsics[..., 0, 0, None, None],
                (grid_y - intrinsics[..., 1, 2, None, None]) / intrinsics[..., 1, 1, None, None],
                torch.ones((*intrinsics.shape[:2], height, width), device=device, dtype=dtype),
            ),
            dim=-1,
        )
        directions_camera = torch.nn.functional.normalize(directions_camera, dim=-1)
        rotation = camera_to_base[..., :3, :3].to(dtype=dtype)
        # Avoid batched GEMM here.  On the target A800/CUDA 12.9 stack,
        # ``einsum``/``bmm`` with a batch dimension >1 selects a strided-GEMM
        # path that raises ``CUBLAS_STATUS_INVALID_VALUE``.  This explicit
        # row-vector @ R^T expansion is equivalent and stays elementwise.
        directions_base = torch.stack(
            (
                directions_camera[..., 0] * rotation[..., 0, 0, None, None]
                + directions_camera[..., 1] * rotation[..., 0, 1, None, None]
                + directions_camera[..., 2] * rotation[..., 0, 2, None, None],
                directions_camera[..., 0] * rotation[..., 1, 0, None, None]
                + directions_camera[..., 1] * rotation[..., 1, 1, None, None]
                + directions_camera[..., 2] * rotation[..., 1, 2, None, None],
                directions_camera[..., 0] * rotation[..., 2, 0, None, None]
                + directions_camera[..., 1] * rotation[..., 2, 1, None, None]
                + directions_camera[..., 2] * rotation[..., 2, 2, None, None],
            ),
            dim=-1,
        )
        origins_base = camera_to_base[..., :3, 3].to(dtype=dtype)[:, :, None, None].expand_as(directions_base)
        return torch.cat((origins_base, directions_base), dim=-1).permute(0, 1, 4, 2, 3).contiguous()

    def _rebuild_plucker_condition(
        self,
        data: dict[str, Any],
        videos: list[torch.Tensor],
        crop: MosaicCropParams,
    ) -> None:
        metadata_keys = (self._CAMERA_TO_BASE_KEY, self._CAMERA_INTRINSICS_KEY, self._CAMERA_SOURCE_SIZE_KEY)
        present = [key in data for key in metadata_keys]
        if not any(present):
            if "plucker_condition" in data:
                raise RuntimeError(
                    "CUDA crop/resize received a prebuilt plucker_condition without camera calibration metadata; "
                    "enable defer_plucker_generation so the ray map can be regenerated from updated intrinsics."
                )
            return
        if not all(present):
            raise ValueError(
                f"Incomplete exterior camera metadata: present={dict(zip(metadata_keys, present, strict=True))}"
            )

        poses = self._metadata_items(data, self._CAMERA_TO_BASE_KEY, (2, 4, 4))
        intrinsics = self._metadata_items(data, self._CAMERA_INTRINSICS_KEY, (2, 3, 3))
        source_sizes = self._metadata_items(data, self._CAMERA_SOURCE_SIZE_KEY, (2, 2))
        if not (len(videos) == len(poses) == len(intrinsics) == len(source_sizes)):
            raise ValueError(
                "CUDA Plucker rebuild requires one calibration per video: "
                f"videos={len(videos)}, poses={len(poses)}, intrinsics={len(intrinsics)}, sizes={len(source_sizes)}"
            )
        pose_batch = torch.stack(poses).to(dtype=torch.float32)
        intrinsic_batch = torch.stack(intrinsics).to(dtype=torch.float32)
        source_size_batch = torch.stack(source_sizes)
        updated_intrinsics = self._update_intrinsics_for_crop_resize(
            intrinsic_batch,
            source_size_batch,
            target_height=crop.bottom_height,
            target_width=crop.left_width,
            crop_box=crop.exterior_box,
        )
        exterior = self._build_batched_plucker_raymaps(
            pose_batch,
            updated_intrinsics,
            height=crop.bottom_height,
            width=crop.left_width,
        )
        condition = exterior.new_zeros((len(videos), 6, crop.height, crop.width))
        condition[:, :, crop.top_height :, : crop.left_width] = exterior[:, 0]
        condition[:, :, crop.top_height :, crop.left_width :] = exterior[:, 1]
        if self.camera_dropout_rate > 0.0:
            drop = torch.rand(len(videos), device=condition.device) < self.camera_dropout_rate
            condition[drop] = 0
        data["plucker_condition"] = [item.unsqueeze(0) for item in condition]
        for key in metadata_keys:
            data.pop(key, None)

    def _drain_timing_events(self, *, synchronize: bool = False) -> None:
        while self._pending_events:
            start, end = self._pending_events[0]
            if synchronize:
                end.synchronize()
            elif not end.query():
                break
            self._pending_events.popleft()
            self._elapsed_ms += float(start.elapsed_time(end))
            self._timed_batches += 1

    def on_train_start(self, model, iteration: int = 0) -> None:
        del model, iteration
        if (self.enabled or self.rebuild_vit_pixel_values) and not self._logged_startup:
            if self.enabled:
                log.info(
                    "GPU image augmentation enabled: shared view-aware crop/resize plus "
                    "torchvision.transforms.v2.ColorJitter on CUDA "
                    f"(crop_scale={self.crop_scale}, chunk_size={self.chunk_size})."
                )
            if self.rebuild_vit_pixel_values:
                log.info(
                    "GPU first-frame ViT rebuild enabled: Qwen pixel_values are rebuilt from the "
                    f"{'augmented' if self.enabled else 'original'} training video on CUDA."
                )
            self._logged_startup = True

    @torch.no_grad()
    def on_training_step_batch_start(self, model, data: dict[str, Any], iteration: int = 0) -> None:
        del model, iteration
        if "video" not in data:
            return
        if not self.enabled and not self.rebuild_vit_pixel_values:
            if "pixel_values" in data:
                pixel_values = self._collect_tensor_leaves(data["pixel_values"], key="pixel_values")
                if any(value.numel() == 0 for value in pixel_values):
                    raise RuntimeError(
                        "Deferred ViT pixel_values reached training with GPU generation disabled; "
                        "enable rebuild_vit_pixel_values or use the CPU ViT processor path."
                    )
            return
        tensors: list[torch.Tensor] = []
        self._flatten_video_tree(data["video"], tensors)
        if not tensors:
            return
        first_device = tensors[0].device
        if first_device.type != "cuda":
            raise RuntimeError(f"GPU image augmentation runs after H2D and expected CUDA input, got {first_device}")
        if any(tensor.device != first_device for tensor in tensors):
            raise RuntimeError("All video tensors in one training batch must be on the same CUDA device")

        self._drain_timing_events()
        stream = torch.cuda.current_stream(first_device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        augmented_tensors = tensors
        if self.enabled:
            spec_tensors: list[torch.Tensor] = []
            video_spec = self._flatten_video_tree(data["video"], spec_tensors)
            augmented_tensors, crop, color_params = self._augment_tensor_list_and_params(
                spec_tensors, require_cuda=True
            )
            data["video"] = self._rebuild_video_tree(video_spec, augmented_tensors)
            if "vit_images" in data:
                vit_contexts = self._collect_tensor_leaves(data["vit_images"], key="vit_images")
                if len(vit_contexts) != len(augmented_tensors):
                    raise ValueError(
                        "GPU ViT augmentation requires one vit_images tensor per video, "
                        f"got {len(vit_contexts)} for {len(augmented_tensors)} videos"
                    )
                updated_contexts = self._augment_vit_contexts(vit_contexts, crop, color_params)
                replacement_index = [0]
                data["vit_images"] = self._replace_tensor_leaves(
                    data["vit_images"], updated_contexts, replacement_index
                )
                if replacement_index[0] != len(updated_contexts):
                    raise AssertionError("Did not consume every augmented ViT context tensor")
            self._rebuild_plucker_condition(data, augmented_tensors, crop)
        if "pixel_values" in data and (self.enabled or self.rebuild_vit_pixel_values):
            if not self.enabled:
                augmented_tensors = []
                self._flatten_video_tree(data["video"], augmented_tensors)
            self._rebuild_vit_pixel_values(data, augmented_tensors)
        end.record(stream)
        self._pending_events.append((start, end))

    def on_training_step_batch_end(
        self,
        model,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, output_batch, loss, iteration
        if self.enabled or self.rebuild_vit_pixel_values:
            self._drain_timing_events()

    def on_training_step_end(
        self,
        model,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, output_batch, loss
        if (
            (not self.enabled and not self.rebuild_vit_pixel_values)
            or not self.log_timing
            or iteration % self.config.trainer.logging_iter != 0
        ):
            return

        # Normal loss/timer logging has already synchronized the training step
        # at this cadence.  Synchronize any unusually delayed event only here,
        # never on every batch.
        self._drain_timing_events(synchronize=True)
        stats = torch.tensor([self._elapsed_ms, float(self._timed_batches)], device="cuda", dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        elapsed_ms, count = stats.tolist()
        if distributed.is_rank0() and count > 0 and wandb.run is not None:
            wandb.log({"timer/gpu_augmentation_ms": elapsed_ms / count}, step=iteration)
        self._elapsed_ms = 0.0
        self._timed_batches = 0

    def on_train_end(self, model, iteration: int = 0) -> None:
        del model, iteration
        if self.enabled or self.rebuild_vit_pixel_values:
            self._drain_timing_events(synchronize=True)
