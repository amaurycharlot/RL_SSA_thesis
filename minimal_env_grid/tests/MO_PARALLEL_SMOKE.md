# Sequential/concurrent MO integration check

Run from the Python project in its training environment:

```powershell
python -m pip install -r requirements-mo-validation.txt
python -B tests/validate_mo_parallel.py check --output runs_mo_validation/parallel_check_001
```

The output directory must not already exist. The check uses only
`tests/fixtures/mo_parallel_smoke.json`, independently of the production config.
It creates `sequential/`, `concurrent/`, per-command logs, `progress.jsonl` and,
only after all checks pass, `evidence.json`. The production configuration is
checked for unchanged content when present. The extra validation dependency,
`psutil`, lets failure/timeout cleanup terminate owned learner process trees.

For unittest discovery:

```powershell
python -B -m unittest discover -s tests -p test_mo_parallel_integration.py -v
```

Unittest normally uses a temporary directory and removes its artifacts. Set
`MO_PARALLEL_SMOKE_OUTPUT` to a new directory to retain its evidence. Keep project
source fixed during a check because ordinary experiment provenance validation
continues to apply. No Slurm installation or scheduler submission is needed.

The fixture declares two ordered training seeds `[9, 2]`, two preferences, two
spawned simulation workers per learner, two evaluation episodes and two holdout
episodes. Each learner has a one-thread numerical limit and a three-CPU
reservation. The concurrent check needs at least six allocated/visible CPUs;
unittest skips it when fewer are available. Inside Slurm, reserve at least six
CPUs for the validation process. Each child then receives its own three-CPU test
reservation. A request for five training transitions rounds to eight at a
four-transition rollout boundary. These are separate smoke settings; the
production eight-worker configuration is never rewritten.

The check executes the existing CLI for a real whole-grid sequential baseline
and a separate initialized grid of policy tasks. Two independent learner
processes hold different policy claims at the same time. While they are paused,
a duplicate policy command and an active-grid finalizer must fail without
changing artifacts. The test-only worker wrapper introduces barriers and one
`KeyboardInterrupt`; model construction, PPO updates, evaluation, checkpointing,
policy claims and final reports use the real implementation.

One policy stops after its committed step-four checkpoint. Other policies finish,
and finalization must refuse the incomplete grid without changing files. Resume
restores step four and learns exactly four more transitions. Completed-task retry
must skip training without changing bytes or file modification times. Workers
must leave shared caches/logs/reports unchanged until the real finalizer runs.

Finally, both experiments are checked against their full saved weights, seed
plans, rounded budgets and committed final bundle identities. Every intermediate
and holdout record must retain its ordered episode seeds, namespace, deterministic
mode, finite objective returns and scaling. Derived final results must be complete
with one hypervolume per training seed, recomputed from that seed's own policy
set. Report manifests must cover both budgets and seeds and match output hashes.
Successful concurrent execution must publish `FINALIZED.json` and release claims.

Comparison covers scientific identity, seed/budget contracts and completeness.
It does not require matching reward vectors, model bytes, timestamps, UUIDs or
cross-hardware bitwise results. Checkpoint continuation also does not restore the
live simulation state for exact environment/RNG replay. The small fixture checks
orchestration and recovery; it does not establish scientific performance,
production-scale throughput or target-cluster behavior.
