#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One Slurm task owns one learner and eight spawned simulation workers.
# The submission helper supplies the throttled array and afterok dependency.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'Usage: policy.sh ABS_PYTHON ABS_LAUNCHER ABS_MANIFEST' >&2
    exit 2
fi
exec "$1" -s -B "$2" run --manifest "$3" --stage policy
