# SAGE algorithm design

## 1. Problem

At routing time an incumbent agent \(a_0\) is already executing task \(x\). The router chooses a mode \(m\) and executor set \(S\):

- \(m=\text{SELF}\), \(S=\{a_0\}\)
- \(m=\text{COLLABORATE}\), \(a_0\in S\), \(|S|>1\)
- \(m=\text{HANDOFF}\), \(S=\{a_j\}\), \(a_j\ne a_0\)

The task contains weighted capability requirements, value, budget, deadline, permissions, risk tolerance, and incumbent progress.

## 2. Hard eligibility

Before learning or ranking, SAGE removes an agent if it is unavailable, lacks a required permission, violates the deadline, or produces an unaffordable bid. This keeps authorization separate from prediction: a high model score cannot grant access.

## 3. Trust-calibrated capability

For agent \(a\), requirement \(r\), advertised capability \(s_{a,r}\), bid confidence \(b_a\), and empirical reliability posterior \(\theta_a\):

\[
\tilde b_a=\mathbb{E}[\theta_a]b_a +(1-\mathbb{E}[\theta_a])0.5
\]

\[
q_{a,r}=s_{a,r}\,(0.5+0.5\mathbb{E}[\theta_a])\,(0.5+0.5\tilde b_a)
\]

Cold-start agents remain selectable, but unsupported confidence claims are shrunk toward 0.5.

## 4. Requirement-graph coverage

For team \(S\), noisy-OR coverage of requirement \(r\) is:

$$
C_r(S)=1-\prod_{a\in S}(1-q_{a,r})
$$

Weighted total coverage is \(C(S)=\sum_r w_rC_r(S)/\sum_rw_r\). Minimum requirement thresholds produce a bottleneck factor, so a team cannot hide one missing critical skill behind high average coverage.

The noisy-OR set function is monotone and submodular under fixed non-negative capabilities. SAGE exploits its diminishing returns with greedy collaborator selection.

## 5. Complementarity and learned synergy

Skill-vector cosine similarity estimates redundancy. Pairwise Beta posteriors estimate whether two agents work well together. The success estimate is:

$$
\hat p=\operatorname{clip}(C(S)B(S)+\eta G(S)-\rho D(S),0,1)
$$

where \(B\) is the critical-requirement bottleneck, \(G\) is centered pair synergy, and \(D\) is average skill redundancy.

## 6. State-aware utility

The utility of a feasible route is:

$$
U=V\hat p-\lambda_c\bar C-\lambda_l\bar L-\lambda_rR-\lambda_hH-\lambda_oO
$$

- \(\bar C\): cost normalized by task budget.
- \(\bar L\): latency normalized by deadline.
- \(R\): empirical failure risk.
- \(H=p_x\cdot h_x\): handoff loss, increasing with incumbent progress \(p_x\).
- \(O=(|S|-1)o_x\): coordination overhead.

This produces a useful asymmetry: an expert may receive a fresh task via HANDOFF, while the same expert may be invited through COLLABORATE after the incumbent has accumulated valuable context.

## 7. Greedy team construction

Start from the incumbent. Repeatedly add the feasible candidate with the largest positive marginal utility until no candidate improves utility or the collaborator limit is reached. SELF and every feasible single-agent HANDOFF are evaluated separately, so collaboration must beat both.

## 8. Online learning

After execution:

- update each selected agent's Beta reliability posterior with task success;
- update each selected pair's synergy posterior;
- retain the full decision trace for offline calibration and counterfactual replay.

Thompson sampling can be enabled to explore uncertain agents. Deterministic posterior means are the default for reproducible tests.

## 9. Production extensions

The next research steps are context-conditioned posteriors, requirement extraction from task text, graph-neural pair scoring, delayed and partial credit assignment, truthful mechanism design for bids, and constrained contextual-bandit training against offline Sprix traces.
