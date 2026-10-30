# MO training on Slurm: operator runbook

Use this guide after the local [integration check](../../tests/MO_PARALLEL_SMOKE.md)
passes. The [template reference](README.md) lists every resource parameter. Paths,
login host, account and partition below are placeholders; replace them with your
site's values. Commands target a Linux cluster with Bash and a shared filesystem.

## 1. Upload and prepare the shared environment

Upload a new, versioned project directory containing `agents/`, `ssa_env/`,
`configs/`, `scripts/slurm/`, `tests/` and both MO requirements files. Keep local
runs, model databases and plots out of the transfer. For example, from local Bash:

```bash
ssh USER@LOGIN 'mkdir -p /shared/mo/releases/run001'
rsync -av --exclude='__pycache__/' --exclude='runs*/' --exclude='rollouts/' \
  --exclude='plots/' --exclude='*.db' \
  /path/to/minimal_env_grid/ USER@LOGIN:/shared/mo/releases/run001/
```

SFTP/SCP can transfer the same selected files from Windows. Keep `.sh` files with
LF line endings. Use an actual shared project path, not a moving symlink. The
project and environment must be accessible at the same absolute paths from the
submission host and compute nodes. Slurm transfers the batch script, not the
rest of the source or environment ([sbatch reference](https://slurm.schedmd.com/sbatch.html)).

Load the site's supported Python/CUDA modules as appropriate, then create the
environment on shared storage. Provision dependencies **before** initialization:

```bash
export PROJECT=/shared/mo/releases/run001
export PY=/shared/mo/envs/run001/bin/python
export ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION
cd "$PROJECT"
python3 -m venv /shared/mo/envs/run001
"$PY" -m pip install -r /shared/mo/environment-lock-linux.txt
"$PY" -m pip install -r requirements-mo-validation.txt
"$PY" -m pip check
mkdir -p /shared/mo/provenance/run001
"$PY" -m pip freeze --all > /shared/mo/provenance/run001/environment.freeze.txt
"$PY" -V > /shared/mo/provenance/run001/python.txt
```

`environment-lock-linux.txt` is an **owner/site-provided** Linux environment lock,
not a file bundled here. It must supply the training/reporting stack: NumPy,
SciPy, PyTorch, Gymnasium, Stable-Baselines3, Matplotlib and pandas, plus their
dependencies; provide wandb only if external logging is enabled. Select the
Torch build compatible with your CPU/GPU nodes. Do not blindly install a Windows
freeze file on Linux. The bundled runtime requirements add threadpoolctl, and
validation requirements add psutil; neither is a complete training lock.

Local block-9 validation used Python 3.9.13, NumPy 2.0.2, SciPy 1.13.1, Torch
2.5.1 CPU, Gymnasium 1.1.1, SB3 2.7.1, Matplotlib 3.9.4, pandas 2.2.3,
threadpoolctl 3.7.0 and psutil 7.2.2. These are a tested reference, not proof of
Linux/GPU compatibility. Validate the chosen cluster environment, freeze it,
then keep it and the versioned source directory fixed for the run.

Run the separate smoke check on a compute allocation with at least six CPUs.
An interactive compute shell can be requested with the site's supported
[`srun`](https://slurm.schedmd.com/srun.html) options:

```bash
srun --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 \
  --cpus-per-task=6 --mem=8G --time=00:30:00 --pty bash
# In the allocated compute shell:
cd "$PROJECT"
"$PY" -s -B tests/validate_mo_parallel.py check \
  --output /shared/mo/validation/run001
```

Inspect `evidence.json` and logs. The smoke fixture deliberately uses two workers;
production keeps eight. Exit this shell before requesting another allocation.
Add site GPU requests when validating a GPU setup. Memory/time examples throughout
this guide must be adjusted for the site; benchmark before the full sweep.

## 2. Choose resources, freeze configuration, submit

First perform the timing and concurrency pilots in section 6. Select resources
from measured complete-task throughput and memory/GPU capacity. Keep the saved
scientific configuration, eight workers, rollout size, seed grid and objective
settings fixed for the production experiment.

```bash
export CONFIG="$PROJECT/configs/ppo_mo_r6_r2.json"
export EXPERIMENT=/shared/mo/runs/run001
export SUB=/shared/mo/submissions/run001
export LEARNER_THREADS=1 POLICY_CPUS=9 MAX_CONCURRENT=4
cd "$PROJECT"
"$PY" -s -B scripts/slurm/mo_slurm.py submit \
  --config "$CONFIG" --experiment "$EXPERIMENT" --submission-dir "${SUB}_preview" \
  --account "$ACCOUNT" --partition "$PARTITION" \
  --learner-threads "$LEARNER_THREADS" --policy-cpus "$POLICY_CPUS" \
  --max-concurrent "$MAX_CONCURRENT" --policy-memory 8G --policy-time 24:00:00 \
  --dry-run
```

Inspect preview `manifest.json` and `submission.json`. Confirm the Python/project
paths, full grid, resolved resource requests and dependency commands. Production
CPU demand is `learner_threads + 8 * saved_worker_threads`: nine CPUs for one
learner thread and eight one-thread workers, ten for two learner threads.
The current selected configuration has five seeds and eleven weights, hence 55
tasks (`0-54`). The helper always derives this from the input rather than assuming it.

Submit the same settings without `--dry-run`, with the new submission directory:

```bash
"$PY" -s -B scripts/slurm/mo_slurm.py submit \
  --config "$CONFIG" --experiment "$EXPERIMENT" --submission-dir "$SUB" \
  --account "$ACCOUNT" --partition "$PARTITION" \
  --learner-threads "$LEARNER_THREADS" --policy-cpus "$POLICY_CPUS" \
  --max-concurrent "$MAX_CONCURRENT" --policy-memory 8G --policy-time 24:00:00
export MANIFEST="$SUB/manifest.json"
"$PY" -m json.tool "$SUB/submission.json"
```

For GPU policies, add the selected `--gres` to **both** preview and submission;
device selection still comes from the saved configuration. The common partition
must accept initialization without GRES. Configure finalizer CPU/memory/time
separately if the defaults do not fit. Initialization owns a new/empty experiment;
the array waits for its success, and finalization waits for the entire array's
success. `%K` controls simultaneous array elements, and whole-array `afterok`
requires all elements to succeed ([Slurm arrays](https://slurm.schedmd.com/job_array.html)).

Do not launch the sequential whole-grid runner, standalone reporting or another
finalizer against this active experiment. Each task loads the full saved plans;
never create a task-specific sliced config. Source/package changes are refused
by the manifest and checkpoint provenance checks. Restore the recorded versioned
source/environment to resume an existing run; do not rewrite its fingerprints.

## 3. Monitor progress and inspect failures

Read the accepted IDs from `submission.json` and set these variables to their
actual values (not the dry-run placeholder IDs):

```bash
export INIT_JOB=123450 ARRAY_JOB=123451 FINAL_JOB=123452
squeue --array --jobs="$INIT_JOB,$ARRAY_JOB,$FINAL_JOB" \
  --format='%.24i %.10T %.10M %.6D %R'
sacct --array --jobs="$ARRAY_JOB" \
  --format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS
tail -n 40 "$SUB/logs/policy-${ARRAY_JOB}_3.err"
tail -n 10 "$SUB/logs/policy-${ARRAY_JOB}_3.out"
```

`squeue --array` expands queued array elements; `sacct` reports completed jobs and
steps ([squeue](https://slurm.schedmd.com/squeue.html),
[sacct](https://slurm.schedmd.com/sacct.html)). Retain step rows when inspecting
memory: allocation-only queries can omit utilization, and accounting availability
depends on the site. Missing memory values are not evidence of zero memory use.
Check memory for the learner and all workers, not just one process's RSS.

Task mapping is `seed_index, weight_index = divmod(task_id, n_weights)`; the actual
run seed is `saved_config.run_seeds[seed_index]`. For the current eleven-weight
grid, task 17 is `seed_1/w_6`, not run seed 17. Inspect that slot's `OWNER.LOCK`,
`logs/events/`, checkpoints and `COMPLETE.json`. Local events record execution
allocation/device/thread settings, evaluation steps and failure phases. The CLI
success JSON records restored/additional transitions and final bundle path.

Completion-marker counts are a progress hint, not a substitute for finalizer
validation. On partial submission, accepted jobs may still run: read recorded
IDs/commands and scheduler stderr before submitting again. A submission that
returned an unrecognized job ID may also have reached the scheduler; inspect
the queue. Do not rerun initialization into a nonempty experiment.

## 4. Resume failed slots and recover interrupted ownership

Wait until **every original array element is terminal**, including queued/requeued
elements. Inspect its failed IDs, stderr and policy artifacts. Verify the learner
and simulation process tree for any selected slot has stopped. Do not race retries
with an original worker. A handled failure normally releases ownership and resumes
from committed checkpoints automatically.

For an `OWNER.LOCK/<token>.json` left by hard exit, inspect its recorded host, PID,
Slurm job/array IDs and exact token. Only after independently confirming that the
owner's process tree and scheduler job stopped, call the explicit recovery API:

```python
from agents.mo.experiment import load_experiment, task_to_policy
from agents.mo.ownership import recover_policy_claim

saved = load_experiment(experiment_dir)
i, j = task_to_policy(failed_task_id, saved.config)
audit_path = recover_policy_claim(
    saved.root, saved.config, i, j,
    expected_token=inspected_owner_token, owner_stopped=True,
    reason="Verified recorded process tree and Slurm job stopped; operator incident reference",
)
print(audit_path)
```

Fill the three input variables from the inspected run. For a stale
`FINALIZER.LOCK/<token>.json`, use `recover_finalizer_claim(saved.root,
saved.config, expected_token=..., owner_stopped=True, reason=...)` after the same
verification. These APIs record recovery authorization before removing the exact
claim. Neither lock age nor absence of a PID on the login host proves a remote
owner stopped. Tokenless/empty claims need offline inspection with all writers and
submitters stopped; do not guess a token or remove an active claim.

Lower-level `RESUME.LOCK` and evaluation `step_*.LOCK` are separate, zero-byte
**files**, not ownership directories. After stopping all writers/submitters and
backing up the experiment, inspect the specific interrupted lock under the
affected results slot. Only the exact inspected regular, nonsymlink, zero-byte
lock file may be unlinked. Do not bulk-delete locks, remove resume
journals/committed records, or fabricate a completion marker. A nonempty,
misplaced or ambiguous lock needs investigation before retry.

The function below prints a retry/finalizer command from the original manifest
and writes a new review record. It **does not submit it**. Define it once in your
operator shell. Review and run the printed command separately; append the actual
accepted job ID to your incident/submission record.

```bash
plan_job() {
  "$PY" -s -B - "$MANIFEST" <<'PY'
import json, os, re, runpy, shlex, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
m = json.loads(manifest_path.read_text())
api = runpy.run_path(str(Path(m['project']) / 'scripts/slurm/mo_slurm.py'))
api['verify'](m)
action = os.environ['MO_ACTION']
dependency = os.environ.get('MO_DEPENDENCY') or None
if dependency is not None and not re.fullmatch(r'[1-9][0-9]*', dependency):
    raise ValueError('Use an actual numeric dependency job ID')
if action == 'retry':
    text = os.environ['MO_TASKS']
    if not re.fullmatch(r'[0-9]+(?:,[0-9]+)*', text):
        raise ValueError('MO_TASKS must be comma-separated numeric saved task IDs')
    tasks = sorted(set(int(value) for value in text.split(',')))
    if any(value >= m['n_tasks'] for value in tasks):
        raise ValueError('Retry task is outside the saved grid')
    command = api['sbatch_command'](m, 'policy', dependency)
    selector = ','.join(map(str, tasks)) + '%' + str(m['max_concurrent'])
    command = ['--array=' + selector if arg.startswith('--array=') else arg for arg in command]
elif action == 'finalize':
    command = api['sbatch_command'](m, 'finalize', dependency)
else:
    raise ValueError('MO_ACTION must be retry or finalize')
record = {'manifest': str(manifest_path), 'action': action, 'command': command}
with Path(os.environ['MO_RECORD']).open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2)
print(shlex.join(command))
PY
}
export MO_ACTION=retry MO_TASKS='3,17' MO_DEPENDENCY=''
export MO_RECORD="$SUB/retry-plan-001.json"
plan_job
```

Replace the example task IDs with inspected failures. This prints a sparse array
over the **original** task IDs and keeps every other scientific/resource setting.
For timeout/OOM retries, a reviewed increase to printed `--time`/`--mem` requests
is execution-only. CPU/thread changes require a separately recorded execution
manifest with matching resources; do not casually edit the original manifest.
Completed slots in a retry validate and skip, which is useful for recovery but
invalid as a new timing measurement.

After submitting the printed retry command, record its actual array job ID, then
print a **new** finalizer command that waits for retry success:

```bash
export RETRY_ARRAY_JOB=123460
export MO_ACTION=finalize MO_DEPENDENCY="$RETRY_ARRAY_JOB"
export MO_RECORD="$SUB/finalizer-plan-002.json"
plan_job
```

The original failed dependency does not become usable after a successful requeue;
the new finalizer must name the retry array. Failed dependencies remain failed
according to [sbatch](https://slurm.schedmd.com/sbatch.html). All other original
slots must already be complete. The finalizer independently validates the full
grid and refuses remaining incomplete/active policies.

## 5. Finalize, verify and retain results

The normal workflow submits finalization automatically. If only final evaluation
or reporting failed, recover a stale finalizer claim if required, confirm there
are no active writers, then print a standalone retry using `plan_job`:

```bash
export MO_ACTION=finalize MO_DEPENDENCY=''
export MO_RECORD="$SUB/finalizer-plan-003.json"
plan_job
```

Review and submit the printed command. This reuses committed holdout records and
regenerates derived outputs. It does not retrain policies. Never run an active
whole-grid runner or report generator beside it.

Check finalizer stderr/exit status and success JSON (`status: finalized`). Retain
`FINALIZED.json`, `reports/report_manifest.json`, the report files, saved config,
weights/seed/budget plans, provenance, policy completion markers, final bundles,
canonical intermediate/holdout records, event logs and submission records.
`FINALIZED.json` is published only after validated inputs, reporting and logger
cleanup. A repeated finalization archives an older success marker first; an
interrupted rebuild may therefore leave no current marker. Do not manually
create one. Back up completed artifacts according to the site's retention policy.

## 6. Time a policy and pilot concurrency before the full sweep

Use a new **benchmark-only** config and experiment for each resource candidate.
Keep the full seed/preference grid and eight-worker scientific settings; shorten
only its declared training/evaluation/checkpoint schedule. For example, this
creates a two-update config in a new external benchmark directory:

```bash
export CASE=/shared/mo/benchmarks/learner1_case001
mkdir -p "$(dirname "$CASE")"
mkdir "$CASE"
"$PY" -s -B - "$CONFIG" "$CASE/config.json" <<'PY'
import json, sys
from pathlib import Path

values = json.loads(Path(sys.argv[1]).read_text())
rollout = values['ppo']['n_envs'] * values['ppo']['n_steps']
budget = 2 * rollout
values.update(requested_timesteps_per_policy=budget, eval_interval=budget, checkpoint_interval=budget)
with Path(sys.argv[2]).open('x', encoding='utf-8') as stream:
    json.dump(values, stream, indent=2)
print({'rollout_size': rollout, 'benchmark_transitions_per_policy': budget})
PY
"$PY" -s -B scripts/slurm/mo_slurm.py submit \
  --config "$CASE/config.json" --experiment "$CASE/experiment" \
  --submission-dir "$CASE/submission" --account "$ACCOUNT" --partition "$PARTITION" \
  --learner-threads 1 --policy-cpus 9 --max-concurrent 2 --dry-run
"$PY" -s -B -m agents.train.train_ppo_mo \
  --initialize --config "$CASE/config.json" --output "$CASE/experiment"
```

The dry run creates the benchmark manifest without submitting its full array or
finalizer. Initialization is passive and can run on the submission host if site
policy permits. Current production rollout size is 4096, so this declares 8192
transitions per benchmark policy. Episode counts, weights, seeds and model/PPO
settings remain as selected; evaluation cost still contributes to timing.

Request a compute shell for the candidate (load the same modules first):

```bash
srun --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 \
  --cpus-per-task=9 --mem=8G --time=01:00:00 --pty bash
# In the compute shell; exported PROJECT, PY and CASE must be present:
cd "$PROJECT"
/usr/bin/time -v -o "$CASE/task0.time.txt" \
  "$PY" -s -B -m agents.train.train_ppo_mo \
  --experiment "$CASE/experiment" --task-id 0 --learner-threads 1 --cpus-per-task 9 \
  > "$CASE/task0.out" 2> "$CASE/task0.err"
```

Read actual/additional transitions from task JSON, numerical/device allocation
from policy events, elapsed time/RSS from timing output and job/step accounting
where available. Divide **additional transitions** by total task wall time to
report end-to-end transitions/second, including worker startup, evaluation and
checkpoint I/O. This is not optimizer-only throughput. Two updates are a quick
startup-inclusive probe; use a longer separate benchmark to estimate steady
state and evaluation/checkpoint cost. Record hardware, node placement, device,
thread counts, seeds, memory, concurrent policies, source/environment versions
and any resume status. Use fresh slots/cases; a completed skip is not a benchmark.

Exit the interactive allocation. To pilot concurrent task throughput, prepare a
**fresh** case using the same setup and manifest (do not reuse completed task 0).
Set `MANIFEST` to that benchmark manifest, define/use `plan_job` above, set
`MO_ACTION=retry`, `MO_TASKS='0,1,2,3'`, empty `MO_DEPENDENCY`, and `MO_RECORD` to a
new plan file. Review/submit the printed four-task array. This is a partial
benchmark grid; do not finalize it as a complete experiment. Repeat with fresh
cases and manifest `--max-concurrent` values 1, 2 and 4 as capacity permits.

Compare learner thread/CPU candidates `(1,9)`, `(2,10)`, `(4,12)` with saved worker
threads still one, on comparable hardware. Change both benchmark manifest and
compute allocation for each candidate. These are experiments to measure, not a
claim that more learner threads improve throughput. Estimate initial concurrency
as `floor(usable CPUs / CPUs per policy)`, then also enforce measured combined
memory and GPU capacity. Choose aggregate completed policies/hour (or summed
additional transitions over the group wall-time window), not just the fastest
single policy. Include storage contention, queue limits and slower-policy spread.

For the full sweep, restore `CONFIG`/`MANIFEST` to the frozen production paths,
choose a new experiment/submission directory, and apply the measured resource
settings in section 2. The smoke and benchmark budgets are never production
budget replacements. No cluster timings or full-sweep speedup are established
by the local tests; record pilot results before committing the full allocation.
