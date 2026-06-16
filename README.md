# OpenClaw-RL: Reproduction

Reproduction of **OpenClaw-RL: Train Any Agent Simply by Talking**
([arXiv 2603.10165](https://arxiv.org/abs/2603.10165)) — Wang et al., Princeton, 2026.

## Core Idea

Every agent interaction produces a **next-state signal** (user reply, terminal stdout, GUI diff, test verdict).
OpenClaw-RL treats these signals as the universal training source for a single policy, via:

- **Evaluative signal** → scalar reward from a PRM judge → GRPO update
- **Directive signal** → corrective hints extracted from next-state → OPD (On-Policy Distillation) update

Four fully decoupled async loops (policy server, environment, PRM judge, Megatron trainer) run in parallel so inference is never blocked by training.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   OpenClaw-RL                       │
│                                                     │
│  ┌──────────────┐       ┌──────────────────────┐   │
│  │ Policy Server│◄─────►│   Environment Server  │   │
│  │  (SGLang)    │       │ terminal/GUI/SWE/tool │   │
│  └──────┬───────┘       └──────────┬───────────┘   │
│         │  (aₜ, sₜ₊₁)             │                │
│         ▼                          ▼                │
│  ┌──────────────┐       ┌──────────────────────┐   │
│  │  PRM / Judge │       │  Async Sample Buffer │   │
│  │   Server     │──────►│        ℬ             │   │
│  └──────────────┘       └──────────┬───────────┘   │
│                                    │                │
│                          ┌─────────▼──────────┐    │
│                          │  Trainer (Megatron) │    │
│                          │   GRPO + OPD loss   │    │
│                          └────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
openclaw-rl/
├── openclaw_rl/
│   ├── policy/        # Policy server (SGLang inference API)
│   ├── environment/   # Environment servers: terminal, GUI, SWE, tool-call
│   ├── judge/         # PRM judge server (evaluative + directive signals)
│   ├── trainer/       # Megatron trainer: GRPO + OPD losses
│   ├── buffer/        # Async sample buffer
│   └── utils/         # Shared utilities
├── configs/           # Hydra configs per experiment
├── scripts/           # Launch scripts
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## Training Objective

The hybrid loss per sample:

```
ℒ = w_RL · ℒ_GRPO + w_OPD · ℒ_OPD
```

**GRPO** uses scalar PRM reward `r ∈ {+1, −1, 0}` with group-standardized advantages.

**OPD** selects the best directive hint `h*` via overlap-guided selection:

```
h* = argmax_h Σᵢ |top-k(π_old(·|s,y<i)) ∩ top-k(π_teacher(·|sʰ,y<i))|
```

then applies a clipped per-token distillation loss over the student's support set.

## Reproduction Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure experiment
cp configs/default.yaml configs/my_experiment.yaml
# edit model paths, GPU allocation, environment endpoints

# 3. Launch all servers
bash scripts/launch_policy_server.sh
bash scripts/launch_env_server.sh   # pick: terminal / gui / swe / tool
bash scripts/launch_judge_server.sh
bash scripts/launch_trainer.sh

# 4. Run training
python scripts/train.py --config configs/my_experiment.yaml
```

## Key Hyperparameters

| Parameter | Personal Agent | General Agent |
|-----------|---------------|---------------|
| Learning rate | 1e-5 | 1e-6 |
| Clip ε (lo/hi) | — | 0.2 / 0.28 |
| OPD clip C | 1 | 2 |
| Top-k width K | 4 | 4 |
| w_RL / w_OPD | 1.0 / 1.0 | 1.0 / 1.0 |
| KL coeff β | — | 0.01 |
| Max response len | 8192 | 8192 |
| Max context len | 16384 | 16384 |

## Paper

```bibtex
@article{wang2026openclawrl,
  title   = {OpenClaw-RL: Train Any Agent Simply by Talking},
  author  = {Yinjie Wang and Xuyang Chen and Xiaolong Jin and Mengdi Wang and Ling Yang},
  journal = {arXiv preprint arXiv:2603.10165},
  year    = {2026}
}
```
