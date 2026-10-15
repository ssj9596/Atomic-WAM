# Atomic-WAM RoboLab reproduction and results

This package contains inference-only RoboLab code, the WAM export and Reasoner LoRA, and the complete 10-repeat results for both instruction variants.

## Public layout

- `cosmos-framework/`: policy server code and checkpoint files.
- `robolab/`: RoboLab evaluation code. Benchmark assets are supplied separately through `ROBOLAB_ASSETS`.
- `robolab/output/default/repeat_0` through `repeat_9`: ten complete default-instruction runs.
- `robolab/output/vague/repeat_0` through `repeat_9`: ten complete vague-instruction runs.

Each repeat has 120 task directories and `episode_results.jsonl`; transient merge manifests, task split files, logs, GPU shards, and local absolute paths were removed.

The aggregate uses 1,200 episodes per instruction variant. Competency denominators are Relational 420, Visual 840, Procedural 220, and Overall 1,200.

## Aggregate metrics

| Instruction | Successful episodes | Overall SR | Overall score | Relational SR / score | Visual SR / score | Procedural SR / score |
|---|---:|---:|---:|---:|---:|---:|
| Default | 475/1,200 | 39.6% | 53.7 | 45.0% / 53.9 | 34.0% / 49.2 | 25.9% / 45.7 |
| Vague | 379/1,200 | 31.6% | 46.7 | 38.6% / 50.0 | 26.3% / 42.1 | 15.9% / 38.2 |

Scores are the mean subtask progress multiplied by 100. The detailed per episode records remain in each repeat's `episode_results.jsonl`.
