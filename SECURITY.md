# Security Policy

## Supported versions

Sprix SAGE Router is currently an early-stage research preview. Security fixes are applied to the latest code on the `main` branch.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Use GitHub's private vulnerability reporting for this repository when available. If that channel is unavailable, contact the maintainer through the public profile and request a private reporting channel without including exploit details in the first message.

Please include:

- the affected component and revision;
- reproduction steps or a minimal proof of concept;
- expected impact, including permission or task-routing consequences;
- suggested mitigations, if known.

## Security scope

The reference implementation does not authenticate agents, sign Agent Cards, transmit A2A tasks, isolate execution, store credentials, or enforce network policy. Production adopters must add identity, authorization, secure transport, secret management, sandboxing, audit logs, abuse controls, and human approval for high-impact actions.

`sprix_a2a.py` parses public card fields and produces a transport-neutral execution plan. It does not make a card trustworthy, verify a signature, connect to its URL, or authorize the selected agent.

## Threat model checklist

| Threat | Reference behavior | Required production control |
|---|---|---|
| Forged identity or capability | Card claims are treated as declarations | Authenticated registry, signature verification, and provenance |
| Missing authorization | Required permissions are a hard filter | Tenant-aware policy engine and least-privilege credentials |
| Deceptive cost or latency bid | Quote fidelity is updated from outcomes | Signed quotes, caps, settlement checks, and dispute handling |
| Prompt or artifact injection | Content is outside the router's scope | Isolation, content scanning, output validation, and DLP |
| Malicious or failed executor | Failed agents can be excluded on replanning | Sandboxing, cancellation, retries, circuit breakers, and approval |
| Poisoned learning evidence | Evidence updates online beliefs | Evaluator authentication, immutable logs, rate limits, and rollback |
| Snapshot tampering or leakage | Snapshots contain learned parameters | Tenant isolation, integrity protection, encryption, and access logs |
| Denial of service | Beam search is bounded by configuration | Candidate, DAG, team-size, time, and request-rate limits |

## Sensitive data

Do not place secrets, access tokens, private prompts, personal data, or raw proprietary artifacts in Agent skill metadata, task identifiers, routing explanations, traces, or state snapshots. Treat all remote descriptions, bids, messages, and artifacts as untrusted input.
