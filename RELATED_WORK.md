# Related work and scope

SAGE combines ideas from several established areas. The reference
implementation is an engineering and research artifact; it does not claim that
bounded team search, contextual learning, or task allocation originated here.

## Comparison by problem setting

| Area | Representative work | What it establishes | How SAGE differs |
|---|---|---|---|
| Coalition formation | [Shehory and Kraus, 1998](https://doi.org/10.1016/S0004-3702(98)00045-9) | Bounded methods for forming agent coalitions under resource constraints | SAGE makes a runtime SELF/COLLABORATE/HANDOFF choice and jointly assigns a requirement DAG; it does not provide coalition-stability or optimality guarantees. |
| Task allocation | [Gerkey and Mataric, 2004](https://doi.org/10.1177/0278364904045564) | A formal taxonomy for single/multi-task robots and single/multi-robot tasks | SAGE targets software agents with permissions, context transfer, bids, and learned evidence rather than physical robots. |
| Combinatorial allocation | [Sandholm, 2002](https://doi.org/10.1016/S0004-3702(01)00159-X) | Exact winner determination and search structure for combinatorial auctions | SAGE uses bounded heuristic beams and quoted resources; it is not an auction mechanism and makes no truthfulness, equilibrium, or global-optimality claim. |
| Contextual bandits | [Li et al., 2010](https://doi.org/10.1145/1772690.1772758) | Online action selection and offline evaluation under contextual feedback | SAGE currently uses a lightweight online predictor and optional Thompson-style exploration, but does not yet log propensities or implement unbiased off-policy evaluation. |
| LLM routing | [RouteLLM, 2025](https://arxiv.org/abs/2406.18665) | Preference-trained routing between stronger and weaker language models | SAGE selects execution mode, agent team, role assignment, and schedule under hard constraints rather than selecting one model endpoint. |
| Dynamic agent networks | [DyLAN, 2024](https://openreview.net/forum?id=XII0Wp1XA9), [GPTSwarm, 2024](https://arxiv.org/abs/2402.16823), [AFlow, 2025](https://openreview.net/forum?id=z5uVAKwmjf) | Dynamic team selection and optimization of agent/workflow graphs | SAGE emphasizes mid-execution state, an already structured requirement DAG, permission-first feasibility, resource quotes, and auditable alternatives. |
| Multi-agent routing | [MasRouter, 2025](https://arxiv.org/abs/2502.11133) | Learned routing over collaboration mode, roles, and model choices | SAGE is a dependency-free constrained reference router with online evidence updates; the repository does not yet provide a comparable trained model or real benchmark. |

## Current evidence gap

The repository benchmark is a deterministic held-out synthetic environment. It
now includes random and greedy team baselines, structurally different evaluator
equations, initialization and exploration ablations, and early/late learning
windows. That makes it more useful for regression and falsification, but it is
still not a comparison on real heterogeneous A2A endpoints.

A publishable study should define the relationship to coalition formation and
task-allocation objectives formally, compare against learned and optimization
baselines on public or auditable task traces, report routing latency and resource
use, and use logged propensities or randomized trials for causal learning claims.
