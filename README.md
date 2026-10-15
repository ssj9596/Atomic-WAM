<div align="center">

<h1>Atomic-WAM: Execution-Grounded Atomic-Motion<br>Reasoning for World Action Models</h1>

Shijun Shi<sup>1*†</sup>, Zhe Liu<sup>2*</sup>, Zhi Hou<sup>3</sup>, Yinan Chen<sup>1†</sup>, Tao Zhang<sup>1</sup>,<br>
Kai Hu<sup>4‡</sup>, Yong Liu<sup>1</sup>, Jiangning Zhang<sup>1‡</sup>

<sup>1</sup>Zhejiang University &nbsp; <sup>2</sup>University of Hong Kong<br>
<sup>3</sup>ACE Robotics &nbsp; <sup>4</sup>Jiangnan University

<sup>*</sup>Equal contribution &nbsp; <sup>‡</sup>Equal advising<br>
<sup>†</sup>Work completed during an internship at ACE Robotics

[**Project Page**](https://ssj9596.github.io/Atomic-WAM-project/) · [**Paper**](https://arxiv.org/abs/2610.20157) · [**Weights**](https://huggingface.co/datasets/MochunniaN1/Atomic-WAM_Robolab)

</div>

## Updates

- **2026.10** — Paper, inference code, and model weights released.
- **2026.10** — Atomic-WAM reproduced by the RoboLab organizers using our released weights. [Official leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html).

## Release plan

- [x] Paper and project page
- [x] WAM weights and Reasoner LoRA
- [x] Inference code and RoboLab evaluation
- [ ] Training code
- [ ] Atomic-motion annotations and data preparation

## Dependencies and installation

The model server runs on Linux with an NVIDIA CUDA GPU. RoboLab additionally requires IsaacSim 5.0.0, IsaacLab 2.2.0, and the official benchmark assets. The installer creates separate environments for the server (Python 3.12) and simulator (Python 3.11), using the included dependency lockfiles and CUDA 12.8 packages.

### 1. Clone the repository

```bash
git clone https://github.com/ssj9596/Atomic-WAM.git
cd Atomic-WAM
```

### 2. Install the environments

Install `uv` and make sure `ffmpeg` and `ss` are available on your system (`ffmpeg` and `iproute2` on Ubuntu).

```bash
python3 -m pip install "uv>=0.11.3"
bash scripts/setup_env.sh
```

To install only the model server, use `bash scripts/setup_env.sh --server-only`. To install only RoboLab, use `--client-only`. The server installation includes the shared dependencies imported by the released policy, even where upstream groups them under training.

### 3. Download the weights

```bash
cosmos-framework/.venv/bin/python scripts/download_weights.py
```

This downloads the WAM, Reasoner LoRA, tokenizer, visual processor, and VAE from [Hugging Face](https://huggingface.co/datasets/MochunniaN1/Atomic-WAM_Robolab). It uses the latest release by default and creates `cosmos-framework/checkpoints`. Existing checkpoint directories are never overwritten. If you already downloaded the release, you can instead link its checkpoint directory:

```bash
ln -s /absolute/path/to/Atomic-WAM_RoboLab_repro/cosmos-framework/checkpoints cosmos-framework/checkpoints
```

### 4. Prepare RoboLab assets

Download the official assets following the [RoboLab instructions](https://research.nvidia.com/labs/srl/projects/robolab/), then set:

```bash
export ROBOLAB_ASSETS=/absolute/path/to/robolab/assets
```

The directory should contain `backgrounds`, `fixtures`, `objects`, `robots`, and `scenes`.

## Quick inference

Start with a single task:

```bash
GPUS=0 EVAL_REPEATS=1 \
TASKS_FILE="$PWD/scripts/smoke_tasks.txt" \
  bash scripts/run_inference.sh
```

The example runs `BananaInBowlTask`: it starts the Atomic-WAM server, connects the RoboLab client, and saves the rollout under `robolab/output/`. The reasoner predicts an atomic motion from the task and visual history, then the WAM predicts the next action chunk.

To try another task, replace the task name in `scripts/smoke_tasks.txt`. Available names are listed in [task_metadata.json](robolab/robolab/tasks/_metadata/task_metadata.json). Stop the run with `Ctrl+C`.

If you already have compatible environments, set `SERVER_PY=/path/to/cosmos/python` and `ROBO_PYTHON=/path/to/robolab/python` before running the same command.

## RoboLab evaluation

Run all 120 tasks with default instructions:

```bash
GPUS=0 EVAL_REPEATS=10 INSTRUCTION_TYPE=default \
  bash scripts/run_inference.sh
```

For vague instructions:

```bash
GPUS=0 EVAL_REPEATS=10 INSTRUCTION_TYPE=vague \
  bash scripts/run_inference.sh
```

Use `GPUS=0,1` to distribute tasks across two GPUs. Each GPU runs an independent policy server; this does not split a single model across GPUs. Unset `TASKS_FILE` if it was exported for a previous single-task run.

The launcher uses the released inference settings, including wrist rotation by 180°, 544×640 input resolution, Reasoner history, adapter-mode reasoning, and previous-frame-relative gripper decoding. See the [evaluation launcher](robolab/scripts/run_atomic_wam_eval.sh) for the configuration and [released results](docs/reproduction-results.md) for the ten-repeat summaries. Weights and recordings are hosted on Hugging Face rather than in this repository.

## Acknowledgments

Our implementation builds on [NVIDIA Cosmos](https://github.com/NVIDIA/cosmos) and [RoboLab](https://research.nvidia.com/labs/srl/projects/robolab/). We thank their authors for releasing the models, code, and benchmark.

## Citation

If you find our work useful, please cite:

```bibtex
@article{shi2026atomicwam,
  title={Atomic-WAM: Execution-Grounded Atomic-Motion Reasoning for World Action Models},
  author={Shi, Shijun and Liu, Zhe and Hou, Zhi and Chen, Yinan and Zhang, Tao and Hu, Kai and Liu, Yong and Zhang, Jiangning},
  journal={arXiv preprint arXiv:2610.20157},
  year={2026}
}
```

## Licenses

Original copyright and license notices are retained. Cosmos files marked `OpenMDW-1.1` are covered by [OpenMDW-1.1](licenses/OpenMDW-1.1.txt); RoboLab files marked `Apache-2.0` are covered by [Apache-2.0](licenses/Apache-2.0.txt). Refer to individual files for applicable notices.
