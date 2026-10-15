# RoboLab inference snapshot

The evaluation entry point is `scripts/run_atomic_wam_eval.sh`. It invokes the
copied multi-GPU server/client launchers and writes RoboLab output under
`robolab/output/` by default. `scripts/repro_run_client.sh` runs
`policies/cosmos3/run.py`, and the copied `assets/` tree is used through the
relocatable `ROBOLAB_ASSETS` setting.

This bundle targets IsaacLab 2.2.0 and IsaacSim 5.0.0. It preserves full
evaluation artifacts, including `episode_results.jsonl`, per-task logs,
videos, and `run_*.hdf5` files. It does not include a Python environment;
provide one externally and set `ROBO_PYTHON` (or `ROBOLAB_PYTHON`) to it.
