# Related work and claim boundary

SAGE does not claim that coalition formation, noisy-OR coverage, marginal team
construction, constrained utility, beam search, Beta beliefs, online logistic
regression, Thompson-style exploration, or communication graphs originated in
this repository.  Its focused hypothesis is that **checkpoint-specific
execution state improves runtime reconfiguration decisions for software
agents**.

## Closest problem settings

| Area | Representative work | Established result | SAGE scope |
|---|---|---|---|
| Multi-skill coalition formation under uncertainty | [Kenari, Jahan, and Jalali, 2011](https://doi.org/10.1109/AISP.2011.5960992) | Multi-skill agents, uncertain partner capability, Bayesian reinforcement learning, and repeated coalition formation | Prior foundation; not a SAGE novelty claim |
| Coalition task allocation | [Service and Adams, 2011](https://doi.org/10.1007/s10458-010-9123-8) | Theory and algorithms for coalition formation in task allocation | Static coalition enumeration and greedy formation are explicit SAGE baselines |
| Resource-constrained coalition formation | [Shehory and Kraus, 1998](https://doi.org/10.1016/S0004-3702(98)00045-9) | Bounded coalition formation methods under resource constraints | SAGE makes no coalition-stability or global-optimality claim |
| Dynamic coalition decision processes | [Coalitional MDP model, 2021](https://ieeexplore.ieee.org/document/9457777) | Dynamic coalition formation can be formulated as a sequential decision problem | Prevents a broad “first dynamic coalition” claim |
| Coalition and communication-edge optimization for agentic AI | [Eslami, 2026](https://arxiv.org/abs/2608.07532) | Marginal-value activation, communication edges as decision variables, submodular special cases, and a Shapley–submodularity bound | SAGE treats DAG edges as schedule output; its distinction is checkpointed execution. That paper explicitly leaves dynamic coalition formation with task progress and outputs as future work |
| Task allocation | [Gerkey and Mataric, 2004](https://doi.org/10.1177/0278364904045564) | Formal taxonomy for single/multi-task robots and single/multi-robot tasks | SAGE targets software-agent artifacts, permissions, bids, and context transfer |
| Combinatorial allocation | [Sandholm, 2002](https://doi.org/10.1016/S0004-3702(01)00159-X) | Exact winner determination and search structure for combinatorial auctions | SAGE uses bounded heuristic search and makes no truthfulness or equilibrium claim |
| Contextual bandits | [Li et al., 2010](https://doi.org/10.1145/1772690.1772758) | Contextual online action selection and offline evaluation | SAGE's online predictor is an implementation choice; it has no regret theorem or unbiased off-policy estimator |
| LLM routing | [RouteLLM, 2025](https://arxiv.org/abs/2406.18665) | Preference-trained routing between stronger and weaker language models | SAGE acts on an in-flight requirement DAG rather than selecting one endpoint before execution |
| Dynamic agent/workflow graphs | [DyLAN, 2024](https://openreview.net/forum?id=XII0Wp1XA9), [GPTSwarm, 2024](https://arxiv.org/abs/2402.16823), [AFlow, 2025](https://openreview.net/forum?id=z5uVAKwmjf) | Dynamic team selection and optimization of agent or workflow graphs | SAGE evaluates continuation, augmentation, and handoff using completed work and artifact reuse |
| Multi-agent routing | [MasRouter, 2025](https://arxiv.org/abs/2502.11133) | Learned routing over collaboration mode, roles, and model choices | SAGE is a dependency-free constrained reference implementation; it does not claim comparable trained-model evidence |

## Narrow claim tested here

At a routing event, SAGE observes completed requirements, current ownership, an
in-flight completion fraction, partial artifact quality, transferability,
failures, and remaining resource constraints.  It compares continuing,
augmenting, and handing off while estimating work that must be redone.  The
controlled progress intervention, checkpoint trajectory replay, and
progress-masked/static-coalition baselines test this claim directly.

This is not a first-of-kind claim.  A publishable related-work review must still
audit dynamic task reallocation, recovery scheduling, checkpoint migration,
workflow replanning, and multi-robot literature beyond the representative
sources above.

## Evidence boundary

The checked-in evaluators are deterministic synthetic environments authored by
the project.  Structural separation from router scoring reduces circularity but
does not create independent external evidence.  Real claims require repeated
checkpointed executions on auditable tasks, heterogeneous endpoints, an
independently governed artifact judge, and uncertainty estimates over runs.
