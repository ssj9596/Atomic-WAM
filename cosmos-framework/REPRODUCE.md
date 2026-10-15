# Cosmos3 inference snapshot

This is the copied Cosmos3 policy-server portion used by the RoboLab
evaluation. The runtime entry point is
`cosmos_framework.scripts.action_policy_server_robolab`; the parent launcher is
`../robolab/scripts/run_atomic_wam_eval.sh`.

The `checkpoints/` directory contains two entries: the exact WAM export
(`wam`), including its local visual processor/tokenizer files and Wan2.2 VAE,
and the Reasoner LoRA export (`reasoner_lora`). Their paths are resolved from
the reproduction root at runtime.

The copied server defaults to the bundled checkpoints. Override
`SERVER_PY`, `COSMOS_SERVER_PYTHON`, `CKPT`, `WAM_CHECKPOINT_PATH`, or
`REASONER_LORA_PATH` when using a different environment or checkpoint.
