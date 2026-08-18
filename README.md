# Sprix SAGE Router

**State-Aware Graph Exchange for A2A task matching**

Sprix SAGE is a research prototype for a decision that ordinary capability discovery does not answer: when an agent is already working on a task, should it continue alone, invite complementary agents, or hand the task off completely?

The router makes an explicit choice among three modes:

- **SELF** — the incumbent agent continues alone.
- **COLLABORATE** — the incumbent keeps ownership and recruits a small complementary team.
- **HANDOFF** — another agent takes full ownership when switching is worth the transfer cost.

It is designed as a decision layer above the [Agent2Agent (A2A) protocol](https://a2a-protocol.org/latest/). A2A supplies discovery, Agent Cards, messages, tasks, artifacts, authentication, and transport; SAGE decides **who should work with whom, in which mode, and why**.

> This repository is an original Sprix algorithm design and executable research prototype. Its combination of tri-mode routing, submodular capability coverage, trust-calibrated bids, state-aware switching cost, and online pairwise learning has not yet been peer reviewed.

## Why another router?

Most routing work chooses one model or one agent before execution. Real A2A systems need a harder mid-execution decision:

1. The current agent has already invested work, so handoff destroys context and progress.
2. A difficult task may need a *complementary team*, not the globally strongest agent.
3. Agent Cards and bids are claims, not calibrated evidence.
4. Permissions, budget, deadline, availability, and coordination overhead are hard constraints.
5. Agent quality and pairwise chemistry change with experience.

SAGE turns those concerns into one auditable utility calculation.

## Algorithm overview

```text
Task + incumbent state + A2A Agent Cards + bids
                     │
        hard eligibility filter
    permissions · budget · deadline · availability
                     │
        ┌────────────┼────────────┐
        │            │            │
      SELF      COLLABORATE    HANDOFF
   incumbent     greedy team    each peer
                  builder
        └────────────┼────────────┘
                     │
      state-aware expected utility
  quality − cost − latency − risk − switching/coordination
                     │
            best feasible route
                     │
     outcome → Bayesian trust + pair-synergy update
```

The central design is a **capability requirement graph**. For requirement \(r\), the coverage of team \(S\) is:

\[
C_r(S)=1-\prod_{a\in S}(1-q_{a,r})
\]

where \(q_{a,r}\) is the agent's declared capability discounted by observed reliability and bid calibration. This noisy-OR form rewards complementary coverage while producing diminishing returns for redundant agents. A greedy team builder therefore gets an efficient approximation to the best budgeted team.

Each route is ranked by:

\[
U(m,S)=V\hat p(\text{success}\mid x,m,S)-\lambda_c C-\lambda_l L-\lambda_r R-\lambda_h H-\lambda_o O
\]

where \(H\) is progress-dependent handoff loss and \(O\) is collaboration overhead. Details are in [ALGORITHM.md](ALGORITHM.md).

## What is unique in SAGE?

- **Mid-execution tri-mode routing.** SELF, COLLABORATE, and HANDOFF compete in one objective instead of being separate rules.
- **Progress-aware delegation.** Handoff becomes harder as the incumbent accumulates useful work; collaboration can preserve that work.
- **Complementarity before prestige.** Team value is marginal requirement coverage plus learned pair synergy, not a leaderboard score.
- **Trust-calibrated bids.** An agent's quoted confidence is shrunk toward a neutral prior according to its observed reliability.
- **Permission-first matching.** Ineligible agents never enter the ranking, regardless of predicted quality.
- **Online adaptation.** Beta posteriors update both individual reliability and pairwise collaboration synergy after every task.
- **Inspectable decisions.** Every result includes probability, cost, latency, risk, coverage, utility, and a natural-language rationale.

## Quick start

Python 3.10+ is sufficient; the prototype has no runtime dependencies.

```bash
python demo.py
python -m unittest -v
python benchmark.py
```

Minimal usage:

```python
from sprix_sage import Agent, Requirement, SAGERouter, Task

agents = [
    Agent("planner", {"planning": 0.92, "coding": 0.55}, cost=0.08, latency_ms=900),
    Agent("coder", {"planning": 0.35, "coding": 0.96}, cost=0.12, latency_ms=1200),
]

task = Task(
    "build-feature",
    requirements=(Requirement("planning", 0.4), Requirement("coding", 0.6)),
    value=1.0,
    budget=0.30,
    deadline_ms=4000,
    progress=0.35,
)

decision = SAGERouter(agents, incumbent_id="planner").route(task)
print(decision.mode, decision.agents, decision.explanation)
```

## A2A integration

Production integration maps A2A data to this prototype as follows:

| A2A / marketplace signal | SAGE input |
|---|---|
| `AgentCard.skills` | normalized capability vector |
| security requirements | `permissions` hard filter |
| supported input/output modes | compatibility filter before routing |
| task status and artifacts | incumbent `progress` and handoff loss |
| provider quote | `Bid(cost, latency, confidence)` |
| completed task evaluation | reliability and synergy posterior update |

The prototype intentionally does not transmit tasks. It returns a routing decision that an A2A client can execute with `message/send`, streaming, task polling, or cancellation.

## Evaluation plan

The included synthetic benchmark is a smoke test, not scientific evidence. A publishable evaluation should compare against self-only, best-single-agent, embedding similarity, RouteLLM-style pairwise routing, and learned MAS routers on:

- GAIA and BrowseComp-style heterogeneous tasks;
- SWE-bench-style long-horizon coding tasks;
- private Sprix marketplace traces with replay;
- cold-start, agent churn, adversarial bids, permission failures, and delayed feedback.

Primary metrics: task success, utility regret, cost, p95 latency, handoff rate, coordination calls, calibration error, and policy violations.

## Research foundations

- [Agent2Agent Protocol Specification](https://github.com/a2aproject/A2A/blob/main/docs/specification.md) — interoperable Agent Cards and stateful tasks.
- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665), ICLR 2025 — cost-quality routing from preferences.
- [A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration](https://openreview.net/pdf?id=XII0Wp1XA9), COLM 2024 — task-specific team selection and dynamic collaboration.
- [GPTSwarm: Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823), ICML 2024 — agent systems as optimizable computation graphs.
- [AFlow: Automating Agentic Workflow Generation](https://openreview.net/attachment?id=z5uVAKwmjf&name=pdf), ICLR 2025 — automated workflow search and optimization.
- [MasRouter: Learning to Route LLMs for Multi-Agent Systems](https://arxiv.org/abs/2502.11133), 2025 preprint — collaboration mode, role, and model routing.

## Status

Research prototype. The scoring model is deliberately lightweight and interpretable; production use requires learned task embeddings, calibrated evaluators, signed Agent Cards, robust identity/reputation, privacy review, and real A2A adapters.

## License

MIT
