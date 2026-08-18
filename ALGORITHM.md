# SAGE v0.2 algorithm design

## 1. Decision problem

An incumbent agent is executing task \(x\). At each routing event, SAGE chooses a mode \(m\), executor team \(S\), requirement-to-agent assignment \(z\), and communication topology \(E\):

- \(m=\mathrm{SELF}\): the incumbent continues alone;
- \(m=\mathrm{COLLABORATE}\): the incumbent retains ownership and recruits peers;
- \(m=\mathrm{HANDOFF}\): one peer receives full ownership.

The decision can be made before execution or after progress, artifacts, failures, and active executors are known. A2A supplies discovery and task transport; SAGE is the policy layer that chooses a feasible execution configuration.

## 2. Task and live state

A task supplies:

- a weighted requirement DAG \(G_x=(R,D)\), including minimum capability thresholds;
- value, budget, deadline, permissions, and risk tolerance;
- coordination, handoff, and replanning friction;
- an estimate of how much incumbent context can be transferred.

`ExecutionState` supplies the active route, completed requirements, current progress, failed agents, transferable context, and consecutive failure count. Completed DAG nodes are removed from the next routing decision while preserving their dependency effects.

## 3. Permission-first feasibility

An agent is removed before ranking when it is unavailable, failed, unauthorized, unaffordable, or unable to satisfy the deadline. Cost and latency quotes are inflated by learned bid-fidelity posteriors and current load.

After a team has been formed, SAGE performs a second feasibility check using total team cost and DAG critical-path latency. A high learned score can never override these constraints.

## 4. Contextual capability calibration

SAGE maintains both a global reliability posterior \(\theta_a\) and a requirement-conditioned posterior \(\theta_{a,r}\). For advertised capability \(s_{a,r}\) and bid confidence \(b_a\):

$$
t_{a,r}=0.35\,\mathbb E[\theta_a]+0.65\,\mathbb E[\theta_{a,r}]
$$

$$
\widetilde b_{a,r}=t_{a,r}b_a+(1-t_{a,r})0.5
$$

$$
q_{a,r}=s_{a,r}(0.65+0.35t_{a,r})(0.70+0.30\widetilde b_{a,r})
$$

This prevents success in one domain from fully transferring to unrelated domains. Cold-start agents remain selectable. With exploration enabled, one coherent Thompson sample is drawn per belief and reused across every candidate compared in the same routing event.

## 5. Team coverage, assignment, and topology

For each remaining requirement, team coverage is:

$$
C_r(S)=1-\prod_{a\in S}(1-q_{a,r})
$$

The requirement is assigned to the team member with the highest calibrated capability. Weighted coverage and the lowest threshold-satisfaction ratio form separate model features, so one missing critical capability cannot be hidden by a high mean.

Requirement dependencies induce communication edges whenever two dependent nodes are assigned to different agents. Independent requirements on different agents can run concurrently; requirements assigned to the same agent are serialized. The resulting resource-constrained DAG schedule estimates critical-path latency before the route is accepted.

Pairwise Beta posteriors model observed collaboration residuals, while skill-vector similarity measures possible redundancy. These are features rather than claims that the complete utility is submodular.

## 6. Learned success model

The original hand-written success equation has been replaced by an online logistic model:

$$
\widehat p(y=1\mid x,m,S,z,E)=\sigma\left(w_0+w^\top\phi(x,m,S,z,E)\right)
$$

Features currently include coverage, bottleneck satisfaction, contextual trust, pair synergy, redundancy, coordination loss, handoff loss, switching loss, and executor load. The model starts from conservative priors and applies regularized stochastic-gradient updates after real outcomes.

This lightweight model is intentionally replaceable. Production deployments can substitute a GBDT, encoder model, Bayesian neural network, or offline contextual-bandit reward model while retaining the same constraint and search layers.

## 7. Progress-aware constrained utility

Each feasible route is ranked by:

$$
U=V\widehat p-\lambda_c\bar C-\lambda_l\bar L-\lambda_rR
  -\lambda_hH-\lambda_oO-\lambda_u\mathcal U+\beta\mathcal B
$$

where:

- \(\bar C\) and \(\bar L\) are budget- and deadline-normalized cost and latency;
- \(R\) is contextual empirical risk;
- \(H\) includes progress-dependent context-transfer loss;
- \(O\) is DAG communication overhead;
- \(\mathcal U\) is posterior uncertainty;
- \(\mathcal B\) is an optional exploration bonus.

When a live route exists, switching loss depends on retained agents, progress, transferable context, and failure count. Replanning therefore becomes easier after repeated failures and harder after valuable non-transferable work has accumulated.

## 8. Bounded beam team search

SAGE evaluates SELF and every feasible single-agent HANDOFF directly. COLLABORATE teams are constructed using bounded beam search:

1. start with the incumbent;
2. expand each frontier team with every eligible peer;
3. reject teams that exceed total budget;
4. evaluate assignment, DAG schedule, probability, and utility;
5. retain the best `beam_width` partial teams;
6. continue until the collaborator limit is reached.

This searches multiple competing team prefixes and can retain an intermediate team even when it is not the single greedy winner. It is still a bounded approximation: SAGE does not claim a global optimum for the non-submodular full objective.

For \(n\) eligible peers, beam width \(B\), collaborator limit \(k\), and \(|R|\) requirements, routing is approximately \(O(Bkn(|R|+k^2))\), excluding candidate retrieval.

## 9. Evidence-aware online updates

`ExecutionOutcome` can contain overall quality, per-agent scores, per-requirement scores, actual cost, and actual latency.

Updates follow the strongest available evidence:

1. explicit agent scores update agent reliability;
2. otherwise, scores of requirements assigned to an agent provide partial credit;
3. if only a team outcome exists, it is treated as low-weight ambiguous evidence;
4. requirement scores update contextual capability posteriors;
5. pair synergy receives residual rather than unconditional full-team credit;
6. quote-versus-actual deviations update cost and latency fidelity;
7. the selected route updates the online success predictor.

This is safer than assigning the same binary outcome to every member, but it is not yet causal credit assignment. Logged propensities, randomized exploration, and doubly robust off-policy evaluation are still needed for production learning.

## 10. External benchmark

`benchmark.py` uses an external nonlinear simulator whose hidden capabilities, pair effects, quality function, realized cost, and realized latency differ from SAGE's prediction model. Its default suite covers five seeds and 2,500 tasks. It compares:

- incumbent-only execution;
- advertised-skill single-agent routing;
- a hidden-information solo oracle;
- static SAGE without outcome updates;
- online SAGE with contextual updates.

The simulator measures external quality, a common quality-cost-latency utility, normalized resource use, deadline violations, and route distribution. It removes the previous circular evaluation in which SAGE's own noisy-OR probability generated its success labels.

The benchmark is still synthetic. Publishable evidence requires real task traces, heterogeneous A2A endpoints, strong learned baselines, calibration and regret analysis, safety tests, and repeated-seed confidence intervals.
