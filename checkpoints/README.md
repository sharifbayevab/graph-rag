# Checkpoints of the frozen system

The system evaluated once on the sealed test (`frozen.json` records every part and its sha256; `SHA256SUMS` lists the files here).

| file | role |
|---|---|
| `router/router.npz` | question router (logistic regression on bge-m3 question vectors): regular / count / version questions |
| `ranker/pi_ref.pt` (+ `pi_ref.json`, `rerank_settings.json`) | RL ranker for regular questions (top-5 list) |
| `agent/agent-s01-grpo-000220.pt` | struct agent, seed s01 (the frozen one) — count questions (Q1, Q1x) |
| `agent/agent-s02-sft.pt`, `agent/agent-s03-grpo-000200.pt` | the same agent, seeds s02 and s03 (reported beside s01) |

Version questions (Q2) use a deterministic rule over the knowledge graph (no checkpoint). Reward weights: F1 0.5, coverage 0.2, diversity 0.2, coherence 0.1, K_max 5.
