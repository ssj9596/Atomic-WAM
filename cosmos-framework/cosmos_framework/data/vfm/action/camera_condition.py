# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Camera-ray conditioning helpers shared by DROID training and RoboLab serving."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def camera_xyzrpy_to_matrix(extrinsics: Sequence[float]) -> torch.Tensor:
    """Convert DROID ``[x, y, z, roll, pitch, yaw]`` camera-to-base pose to 4x4."""
    if len(extrinsics) != 6:
        raise ValueError(f"Expected six camera extrinsic values, got {len(extrinsics)}")
    values = torch.as_tensor(extrinsics, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError(f"Camera extrinsics must be finite, got {list(extrinsics)!r}")

    x, y, z, roll, pitch, yaw = values.tolist()
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    # Equivalent to scipy Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix().
    rotation = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float32,
    )
    camera_to_base = torch.eye(4, dtype=torch.float32)
    camera_to_base[:3, :3] = rotation
    camera_to_base[:3, 3] = torch.tensor([x, y, z], dtype=torch.float32)
    return camera_to_base


def opengl_camera_pose_to_opencv_matrix(position: Sequence[float], quaternion_wxyz: Sequence[float]) -> torch.Tensor:
    """Convert an Isaac/OpenGL camera pose to an OpenCV camera-to-world matrix.

    Isaac camera quaternions use ``(w, x, y, z)`` and an OpenGL optical frame
    (``+x`` right, ``+y`` up, ``-z`` forward). The returned pose uses OpenCV's
    optical frame (``+x`` right, ``+y`` down, ``+z`` forward), matching DROID.
    """
    if len(position) != 3 or len(quaternion_wxyz) != 4:
        raise ValueError("Expected a 3D position and a wxyz quaternion")
    pos = torch.as_tensor(position, dtype=torch.float32)
    quat = torch.as_tensor(quaternion_wxyz, dtype=torch.float32)
    if not torch.isfinite(pos).all() or not torch.isfinite(quat).all():
        raise ValueError("Camera position/quaternion must be finite")
    quat_norm = torch.linalg.vector_norm(quat)
    if float(quat_norm) <= 0.0:
        raise ValueError("Camera quaternion must have non-zero norm")
    w, x, y, z = (quat / quat_norm).tolist()
    rotation_opengl_to_world = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )
    opencv_to_opengl = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
    camera_to_world = torch.eye(4, dtype=torch.float32)
    camera_to_world[:3, :3] = rotation_opengl_to_world @ opencv_to_opengl
    camera_to_world[:3, 3] = pos
    return camera_to_world


def scaled_camera_matrix(
    camera_matrix: Sequence[float],
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> torch.Tensor:
    """Build and resize a pinhole K matrix from DROID's ``[fx,cx,fy,cy]``."""
    if len(camera_matrix) != 4:
        raise ValueError(f"Expected [fx,cx,fy,cy], got {list(camera_matrix)!r}")
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Source and target camera sizes must be positive")
    fx, cx, fy, cy = torch.as_tensor(camera_matrix, dtype=torch.float32).tolist()
    values = torch.tensor([fx, cx, fy, cy], dtype=torch.float32)
    if not torch.isfinite(values).all() or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Camera intrinsics must be finite with positive fx/fy, got {list(camera_matrix)!r}")
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    intrinsic = torch.eye(3, dtype=torch.float32)
    intrinsic[0, 0] = fx * scale_x
    intrinsic[1, 1] = fy * scale_y
    intrinsic[0, 2] = cx * scale_x
    intrinsic[1, 2] = cy * scale_y
    return intrinsic


def build_plucker_raymap(
    camera_to_world: torch.Tensor,
    intrinsic: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return ``[ray_origin, ray_direction]`` with shape ``[6,H,W]``."""
    if camera_to_world.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError(
            f"Expected camera_to_world [4,4] and intrinsic [3,3], got "
            f"{tuple(camera_to_world.shape)} and {tuple(intrinsic.shape)}"
        )
    if height <= 0 or width <= 0:
        raise ValueError(f"Raymap size must be positive, got {height}x{width}")
    device = camera_to_world.device
    dtype = camera_to_world.dtype
    intrinsic = intrinsic.to(device=device, dtype=dtype)
    y_coords = torch.arange(height, device=device, dtype=dtype) + 0.5
    x_coords = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    directions_camera = torch.stack(
        [
            (grid_x - intrinsic[0, 2]) / intrinsic[0, 0],
            (grid_y - intrinsic[1, 2]) / intrinsic[1, 1],
            torch.ones_like(grid_x),
        ],
        dim=-1,
    )
    directions_camera = torch.nn.functional.normalize(directions_camera, dim=-1)
    directions_world = directions_camera @ camera_to_world[:3, :3].transpose(0, 1)
    origins_world = camera_to_world[:3, 3].view(1, 1, 3).expand_as(directions_world)
    return torch.cat([origins_world, directions_world], dim=-1).permute(2, 0, 1).contiguous()


def compose_exterior_plucker_condition(
    left_camera_to_world: torch.Tensor,
    left_intrinsic: torch.Tensor,
    right_camera_to_world: torch.Tensor,
    right_intrinsic: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Compose wrist-zero/top + physical-left/right raymaps with shape ``[6,H,W]``.

    For the current 540x640 DROID concat, the output is ``[6,540,640]``:
    the top ``[6,360,640]`` wrist region is zero, and the bottom contains
    physical-left and physical-right ``[6,180,320]`` raymaps side by side.
    """
    bottom_height = height // 3
    top_height = height - bottom_height
    left_width = width // 2
    right_width = width - left_width
    condition = torch.zeros((6, height, width), dtype=torch.float32)
    condition[:, top_height:, :left_width] = build_plucker_raymap(
        left_camera_to_world, left_intrinsic, height=bottom_height, width=left_width
    )
    condition[:, top_height:, left_width:] = build_plucker_raymap(
        right_camera_to_world, right_intrinsic, height=bottom_height, width=right_width
    )
    return condition


def renormalize_plucker_directions(condition: torch.Tensor) -> torch.Tensor:
    """Renormalize interpolated direction channels while preserving zero regions."""
    if condition.ndim != 3 or condition.shape[0] != 6:
        raise ValueError(f"Expected plucker condition [6,H,W], got {tuple(condition.shape)}")
    directions = condition[3:6]
    norms = torch.linalg.vector_norm(directions, dim=0, keepdim=True)
    condition = condition.clone()
    condition[3:6] = torch.where(norms > 1e-6, directions / norms.clamp_min(1e-6), torch.zeros_like(directions))
    return condition


__all__ = [
    "build_plucker_raymap",
    "camera_xyzrpy_to_matrix",
    "compose_exterior_plucker_condition",
    "opengl_camera_pose_to_opencv_matrix",
    "renormalize_plucker_directions",
    "scaled_camera_matrix",
]
