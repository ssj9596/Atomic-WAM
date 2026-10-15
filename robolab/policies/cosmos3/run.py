# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the Cosmos3 policy backend across registered tasks."""

import argparse
import sys
import traceback

import cv2  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "cosmos3"

parser = argparse.ArgumentParser(description="Evaluate the Cosmos3 policy backend.")
parser.add_argument(
    "--remote-host", default="localhost", help="Remote host for policy server (default: localhost)."
)
parser.add_argument(
    "--remote-port", default=8000, type=int, help="Remote port for policy server (default: 8000)."
)
parser.add_argument(
    "--filter-isolated-gripper-commands",
    action="store_true",
    help="Remove one-step binary gripper command pulses (default: disabled).",
)
parser.add_argument(
    "--relative-gripper-commands",
    action="store_true",
    help=(
        "Interpret the returned gripper action relative to the current observed state "
        "with a dead-band (default: disabled)."
    ),
)
parser.add_argument(
    "--previous-frame-relative-gripper-commands",
    action="store_true",
    help=(
        "Interpret each returned gripper value relative to the preceding value; "
        "only the first value uses the current observation (default: disabled)."
    ),
)
parser.add_argument(
    "--gripper-relative-threshold",
    type=float,
    default=0.10,
    help="Relative gripper dead-band in [0,1] when --relative-gripper-commands is enabled (default: 0.10).",
)
parser.add_argument(
    "--reasoner-history-images",
    action="store_true",
    help=(
        "Send causal 15 Hz history after enough frames are available; concat mode uses "
        "-16/-8/current, while separate-view mode uses PREV(-16)/CURRENT (default: disabled)."
    ),
)
parser.add_argument("--reasoner-separate-views", action="store_true", help="Send separate LEFT/RIGHT/WRIST Reasoner views.")
parser.add_argument(
    "--reasoner-three-group-history",
    action="store_true",
    help="In separate-view mode, send t-16, t-8, and current groups (9 images) instead of t-16/current (default: disabled).",
)
parser.add_argument("--reasoner-image-long-edge", type=int, default=448, help="Long-edge resize for separate Reasoner views.")
parser.add_argument(
    "--reasoner-include-preview-action",
    action="store_true",
    help=(
        "Feed the previous Reasoner subtask into the next three-image request; "
        "also enables causal Reasoner history images (default: disabled)."
    ),
)

from robolab.eval.runner import add_common_eval_args, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.cosmos3.client import Cosmos3Client
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

auto_register_droid_envs(task=args_cli.task, cameras=WRIST_LEFT_RIGHT_HEAD)


def make_client(args: argparse.Namespace) -> Cosmos3Client:
    """ """
    return Cosmos3Client(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        filter_isolated_gripper_commands=args.filter_isolated_gripper_commands,
        relative_gripper_commands=args.relative_gripper_commands,
        previous_frame_relative_gripper_commands=args.previous_frame_relative_gripper_commands,
        gripper_relative_threshold=args.gripper_relative_threshold,
        reasoner_history_images=args.reasoner_history_images,
        reasoner_separate_views=args.reasoner_separate_views,
        reasoner_three_group_history=args.reasoner_three_group_history,
        reasoner_include_preview_action=args.reasoner_include_preview_action,
    )


def main() -> None:
    """ """
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
