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
