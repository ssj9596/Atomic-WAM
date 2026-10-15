# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action dataset wrappers for Cosmos Action.

Concrete dataset modules may depend on optional third-party packages.  Keep this
package initializer lazy so importing one dataset does not require every optional
backend to be installed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ActionBaseDataset",
    "AgiBotWorldBetaLeRobotDataset",
    "BridgeOrigLeRobotDataset",
    "DROIDLeRobotDataset",
    "RoboMINDFrankaDataset",
    "RoboTwinLeRobotDataset",
    "UMILeRobotDataset",
]

_LAZY_IMPORTS = {
    "ActionBaseDataset": "cosmos_framework.data.vfm.action.datasets.base_dataset",
    "AgiBotWorldBetaLeRobotDataset": "cosmos_framework.data.vfm.action.datasets.agibotworld_beta_lerobot_dataset",
    "BridgeOrigLeRobotDataset": "cosmos_framework.data.vfm.action.datasets.bridge_orig_lerobot_dataset",
    "DROIDLeRobotDataset": "cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset",
    "RoboMINDFrankaDataset": "cosmos_framework.data.vfm.action.datasets.robomind_franka_dataset",
    "RoboTwinLeRobotDataset": "cosmos_framework.data.vfm.action.datasets.robotwin_lerobot_dataset",
    "UMILeRobotDataset": "cosmos_framework.data.vfm.action.datasets.umi_lerobot_dataset",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(_LAZY_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
