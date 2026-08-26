# Operations and production-readiness guide

Sprix SAGE Router is a research preview. This guide defines the controls an adopter should add before exposing the router to real agents, budgets, or sensitive data.

## Recommended request lifecycle

1. Authenticate the caller and resolve tenant policy.
2. Validate the task DAG and place explicit caps on requirements, candidates, beam width, and collaborators.
3. Retrieve a bounded candidate set from a trusted registry.
4. Verify permissions and normalize quotes before invoking SAGE.
5. Store the routing trace with a correlation ID and policy version.
6. Require approval for high-risk, destructive, regulated, or high-cost routes.
7. Dispatch through an isolated executor and enforce cancellation deadlines.
8. Evaluate artifacts independently and record granular outcome evidence.
9. Persist a versioned learning snapshot using an atomic write or transactional store.

## Audit record

`RoutingTrace.to_dict()` is JSON serializable and contains:

- the selected mode, team, assignments, topology, utility, and rationale;
- all feasible alternatives ranked by utility;
- every eligible agent;
- excluded agents and hard-filter reasons such as missing permissions, cost, latency, failure, or availability.

Store task and policy identifiers alongside the trace, but do not place secrets, private prompts, or raw credentials inside router objects.

## State persistence

`export_state()` captures contextual reliability, skill beliefs, pair synergy, cost and latency fidelity, and online model parameters. `restore_state()` validates the schema version and requires an exact agent roster before changing live state.

The snapshot does not contain tasks, artifacts, credentials, or the random generator state. Encrypt production snapshots at rest, attach tenant boundaries, and apply concurrency control so two workers cannot silently overwrite newer evidence.

## Metrics to collect

Measure the complete system rather than only SAGE's predicted probability:

- end-to-end task and requirement success;
- artifact quality and human override rate;
- route distribution and switch frequency;
- candidate count and routing compute time;
- actual versus quoted cost and latency;
- deadline misses, retries, cancellations, and recovery success;
- permission denials and policy violations;
- probability calibration, regret, and drift by task segment;
- P50, P95, and P99 latency for routing and execution separately.

## Rollout gates

Start with offline trace replay, then shadow decisions, then a low-risk canary. Keep a deterministic fallback route and a kill switch. A production rollout should not advance until quality, cost, latency, calibration, security, and recovery thresholds are defined from real tasks.
