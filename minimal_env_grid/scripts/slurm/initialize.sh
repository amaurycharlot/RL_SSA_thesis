#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Resource/account/log options are supplied by mo_slurm.py submit.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'Usage: initialize.sh ABS_PYTHON ABS_LAUNCHER ABS_MANIFEST' >&2
    exit 2
fi
exec "$1" -s -B "$2" run --manifest "$3" --stage initialize
