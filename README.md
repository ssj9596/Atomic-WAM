# Atomic-WAM

**Atomic-WAM: Execution-Grounded Atomic-Motion Reasoning for World Action Models**

[Paper](https://arxiv.org/abs/2610.20157) · [Project page](https://ssj9596.github.io/Atomic-WAM-project/) · [Weights](https://huggingface.co/datasets/MochunniaN1/Atomic-WAM_Robolab)

This repository contains the released Atomic-WAM inference code and RoboLab evaluation pipeline. A vision-language reasoner predicts the next atomic motion, which conditions the WAM to generate an action chunk.

Training code and the annotation dataset are under review.

## Contents

- `cosmos-framework/`: policy server, model implementation, and inference configuration.
- `robolab/`: evaluation client, task definitions, and launch scripts.
- Model weights and evaluation recordings remain on Hugging Face.

The inference source is extracted from the [Hugging Face release](https://huggingface.co/datasets/MochunniaN1/Atomic-WAM_Robolab/tree/a701209fa4553daa077a377b6b591eb0e8f4e249/Atomic-WAM_RoboLab_repro), revision `a701209fa4553daa077a377b6b591eb0e8f4e249`. The source files, launch parameters, and lockfiles are preserved.

## Environment

Run inference on Linux with an NVIDIA CUDA GPU. The model server and simulator can use separate Python environments. The included `pyproject.toml` and `uv.lock` files record their dependencies; RoboLab targets Python 3.11+, IsaacLab 2.2.0, and IsaacSim 5.0.0. The server includes CUDA 12.8 and 13.0 dependency groups; select the group compatible with your environment.

Provide an installed Cosmos server environment through `SERVER_PY` and an installed RoboLab environment through `ROBO_PYTHON`. RoboLab benchmark assets are supplied separately through `ROBOLAB_ASSETS`. See the [RoboLab project](https://research.nvidia.com/labs/srl/projects/robolab/) for benchmark access.

## Download weights

From this repository root, download only the checkpoint subtree:

```bash
hf download MochunniaN1/Atomic-WAM_Robolab \
  --repo-type dataset \
  --revision a701209fa4553daa077a377b6b591eb0e8f4e249 \
  --include 'Atomic-WAM_RoboLab_repro/cosmos-framework/checkpoints/*' \
  --local-dir weights

ln -s ../weights/Atomic-WAM_RoboLab_repro/cosmos-framework/checkpoints cosmos-framework/checkpoints
```

The expected checkpoint directories are `cosmos-framework/checkpoints/wam` and `cosmos-framework/checkpoints/reasoner_lora`. The WAM directory also contains the visual processor, tokenizer, and VAE files required by the launch scripts. If you already have the release weights, link that checkpoint directory instead.

## Run RoboLab evaluation

From the repository root:

```bash
SERVER_PY=/path/to/cosmos/environment/bin/python \
ROBO_PYTHON=/path/to/robolab/environment/bin/python \
ROBOLAB_ASSETS=/path/to/robolab/assets \
GPUS=0 EVAL_REPEATS=1 INSTRUCTION_TYPE=default \
  bash robolab/scripts/run_atomic_wam_eval.sh
```

Use `INSTRUCTION_TYPE=vague` for vague instructions. For the released ten-repeat protocol, set `EVAL_REPEATS=10`. `GPUS=0,1` distributes evaluation across two independent server/client pairs; it is not model parallelism.

The entry point retains the release configuration: adapter-mode reasoner, image history, wrist rotation by 180°, 544×640 resolution, JSON motion prompts, and previous-frame-relative gripper handling. Outputs are written beneath `robolab/output/`.

For launch details, see [server notes](cosmos-framework/REPRODUCE.md), [evaluation notes](robolab/REPRODUCE.md), and the [released evaluation results](docs/reproduction-results.md).

## Upstream code

Atomic-WAM builds on NVIDIA Cosmos and RoboLab. Existing copyright and license notices are retained. Cosmos files marked `OpenMDW-1.1` are covered by [OpenMDW-1.1](licenses/OpenMDW-1.1.txt); RoboLab files marked `Apache-2.0` are covered by [Apache-2.0](licenses/Apache-2.0.txt). Refer to individual files for applicable notices.
