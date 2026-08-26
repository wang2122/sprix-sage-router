# Examples

- `a2a_execution_plan.py` normalizes Agent Card skills, produces an auditable routing trace, and converts the winner into a transport-neutral execution plan.
- `replan_and_persist.py` records failure evidence, replans around a failed incumbent, and restores the learned state in a new router instance.

Run from the repository root:

```bash
python -m examples.a2a_execution_plan
python -m examples.replan_and_persist
```

The examples do not contact remote agents or write credentials. Replace the example card URLs and local scores only after your integration has authenticated the provider and calibrated its evidence.
