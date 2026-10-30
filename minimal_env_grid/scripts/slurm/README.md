# Slurm workflow for the MO policy grid

For upload/environment setup, monitoring, failed-slot recovery, finalizer retries
and per-policy/concurrency benchmarking, follow the [operator runbook](RUNBOOK.md).

Use a fixed shared copy of this project and a shared training Python environment
available at the same absolute paths on the submission host and compute nodes.
Install the project's dependencies, including `requirements-mo-runtime.txt`,
before submission. Load any site-required CUDA/compiler modules before
submitting; jobs inherit the submitted environment. Do not edit the shared source,
input configuration or environment until the workflow finishes.

From the project directory, first write a command plan without submitting jobs:

```bash
/shared/venvs/mo/bin/python -s -B scripts/slurm/mo_slurm.py submit \
  --config /shared/mo/configs/ppo_mo_r6_r2.json \
  --experiment /shared/runs/mo_run_001 \
  --submission-dir /shared/slurm/mo_run_001_preview \
  --account YOUR_ACCOUNT --partition YOUR_PARTITION \
  --learner-threads 1 --policy-cpus 9 --max-concurrent 4 \
  --policy-memory 8G --policy-time 24:00:00 \
  --dry-run
```

Inspect `manifest.json` and `submission.json` in the preview directory. To submit,
use the same command without `--dry-run` and with a **new** submission directory,
for example `/shared/slurm/mo_run_001`. Paths in the example are placeholders.
Memory, wall time and concurrency defaults are starting values for measurement,
not measured resource recommendations. Choose values allowed by your site.

The helper submits these three jobs with `sbatch --parsable`:

1. `initialize.sh`: `--initialize --config ... --output ...`, once for the full grid.
2. `policy.sh`: `--array=0-(N-1)%K`, dependent on `afterok:INIT_JOB_ID`.
   Each array task runs exactly one `--task-id "$SLURM_ARRAY_TASK_ID"` from the
   initialized experiment, with one learner and its eight spawned workers.
3. `finalize.sh`: `--finalize --experiment ...`, dependent on
   `afterok:ARRAY_JOB_ID` for the **entire** array.

The array size is derived from the full frozen input configuration:
`N = len(run_seeds) * n_weights`. The current checked-in configuration declares
55 tasks, so its array is `0-54%K`. No task slices or renumbers the saved grid.
Jobs request one node and one Slurm task; simulation workers are child processes,
not additional Slurm tasks. Success dependencies use `--kill-on-invalid-dep=yes`;
a failed initialization or array prevents its dependent job from running.
The whole-array `afterok` behavior and `%K` throttle follow the
[official Slurm array documentation](https://slurm.schedmd.com/job_array.html).

## Resource parameters

| Option | Default | Applies to |
|---|---|---|
| `--account`, `--partition` | Required | All three jobs |
| `--max-concurrent` | `4` | Array concurrency throttle |
| `--learner-threads` | `1` | Learner Torch/BLAS limits |
| `--policy-cpus`, `--policy-memory`, `--policy-time` | `9`, `8G`, `24:00:00` | Each array task |
| `--init-cpus`, `--init-memory`, `--init-time` | `1`, `2G`, `00:10:00` | Initialization |
| `--finalizer-cpus`, `--finalizer-memory`, `--finalizer-time` | `1`, `8G`, `04:00:00` | Finalization |
| `--finalizer-threads` | `1` | Finalizer numerical environment limits |
| `--gres` | None | Optional policy/finalizer GRES, e.g. `gpu:1` |

Policy allocation must fit `learner_threads + 8 * saved_worker_threads`.
For eight one-thread workers, a two-thread learner needs at least ten CPUs.
Submission rejects other worker topologies and insufficient allocations without
changing the scientific configuration. The existing runtime also checks the
actual scheduler allocation and visible CPU affinity on the compute node.

The configured device remains part of the saved experiment; `--gres` only
requests scheduler resources. If using GPU resources, choose a partition/GRES
combination supported by your site. The common partition must also accept the
initialization job, which requests no GRES. Finalization uses its own thread/CPU
allocation and performs no policy training.

## Fixed inputs, logs and failures

The manifest records the absolute shared project, input config and Python paths,
Python version, key numerical-package versions, input config hash, and hashes of
Python sources under `agents/`, `ssa_env/` and `scripts/slurm/`, plus the shell
templates. Every job verifies these before executing the CLI. Added, deleted or
changed source files are refused. Existing experiment/checkpoint provenance
checks still apply. Jobs perform no installation, checkout or scientific edits.
The manifest checks do not replace keeping the shared environment/source fixed.

The helper creates the submission/log directories before `sbatch`; keep them
outside the experiment directory, whose initialization destination must be new or
empty. Log names include `%j` for initialization/finalization and `%A_%a` for
array tasks. Policy artifacts and execution events retain the existing isolated
`models/seed_I/w_J/` layout.

`submission.json` records each exact argument vector and each accepted job ID
immediately after submission. A failure stops further submissions and preserves
accepted IDs and scheduler stderr. Earlier jobs may still run; inspect this
record before deciding whether to cancel or continue them. An unrecognized
`sbatch` response stops the chain; check the queue for a possibly accepted job.
The helper does not automatically cancel accepted jobs or resubmit failed slots.
Use a new submission directory for a new workflow; an initialized experiment
must be resumed through individual task commands, not initialized again.

The `.sh` files are templates driven by the helper's resource/dependency options.
Their positional arguments are the absolute Python executable, launcher script
and manifest; run the submission helper rather than submitting them without
those arguments. Slurm reads the scripts directly, so executable file bits are
not required. Files must retain LF line endings.

See the [official sbatch reference](https://slurm.schedmd.com/sbatch.html) for
site-supported resource syntax, parsable job IDs and dependency options.
