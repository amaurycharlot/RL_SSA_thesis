#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Submitted afterok on the entire policy array, not afterany or one array task.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'Usage: finalize.sh ABS_PYTHON ABS_LAUNCHER ABS_MANIFEST' >&2
    exit 2
fi
exec "$1" -s -B "$2" run --manifest "$3" --stage finalize
