#!/bin/bash
# Submit run_experiments.py as a single Slurm job (one node, all cells run through one multiprocessing pool).
#
# Run it from the repo checkout on the cluster's shared filesystem, after editing SWEEP in run_experiments.py:
#
#   ./run_sweep_slurm.sh                                   # new sweep, 50 CPUs on gaips_phd
#   ./run_sweep_slurm.sh --cpus 32 --time 48:00:00 --base-seed 3 --name highbeta
#   ./run_sweep_slurm.sh --resume data/sweep_demo_...      # finish a sweep interrupted by the time limit
#
# The job copies the repo to node-local scratch and runs from there, so later edits to the checkout don't affect a
# queued or running job. Results are written straight to --data-dir (default: <repo>/data) on the shared filesystem,
# so manifest.json is up to date after every cell and a killed job can be resumed with --resume.
#
# Every job builds its own Python env (--python, default 3.8) on node-local scratch with conda (--conda, default
# $CONDA_EXE or ~/miniconda3/bin/conda), so no conda env needs to exist or be activated. Nothing is installed on /cfs.

set -e

REPO_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_DIR=$REPO_DIR/data
PARTITION=gaips_phd
CPUS=50
TIME=96:00:00
MEM=
PY_ARGS=
PYTHON_VERSION=3.8                                # matches the local setup (requirements.txt pins numpy 1.24.4)
CONDA_BIN=${CONDA_EXE:-$HOME/miniconda3/bin/conda}  # CONDA_EXE is set by an initialized conda shell

while [ $# -gt 0 ]; do
    case $1 in
    --python) PYTHON_VERSION=$2 ; shift ;;
    --conda) CONDA_BIN=$2 ; shift ;;
    --partition) PARTITION=$2 ; shift ;;
    --cpus) CPUS=$2 ; shift ;;
    --time) TIME=$2 ; shift ;;
    --mem) MEM=$2 ; shift ;;              # e.g. 64G; default: the partition's default
    --data-dir) DATA_DIR=$(realpath "$2") ; shift ;;
    --base-seed) PY_ARGS="$PY_ARGS --base-seed $2" ; shift ;;
    --name) PY_ARGS="$PY_ARGS --name $2" ; shift ;;
    --resume) PY_ARGS="$PY_ARGS --resume $(realpath "$2")" ; shift ;;
    -h|--help) sed -n '2,15p' "$0" ; exit 0 ;;
    *) echo "$0: unrecognized option $1" 1>&2 ; exit 1 ;;
    esac
    shift
done

if [ ! -x "$CONDA_BIN" ]; then
    echo "$0: conda not found at $CONDA_BIN (pass --conda /path/to/bin/conda)" 1>&2
    exit 1
fi

MEM_LINE=
if [ -n "$MEM" ]; then
    MEM_LINE="#SBATCH --mem=$MEM"
fi

mkdir -p "$DATA_DIR" "$REPO_DIR/slurm_logs"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=risk-aware-mcts-sweep
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --cpus-per-task=$CPUS           # cpu-cores per task (= run_experiments.py worker processes)
#SBATCH --export=ALL                    # export all environment variables
#SBATCH --time=$TIME                    # total run time limit (HH:MM:SS)
#SBATCH --partition=$PARTITION
#SBATCH --output=$REPO_DIR/slurm_logs/sweep-%j.out
$MEM_LINE

set -e

# Move \$HOME to the job's folder created by slurm
export HOME=/scratch/slurm-jobs/\$SLURM_JOB_ID
chmod -R 700 \$HOME  # group and others can't read / write / execute
cd \$HOME

# Copy the repo (with .git, so sweep_config.json records the commit), without results or old venvs.
mkdir RiskAwareMCTS
tar -C $REPO_DIR --exclude=./data --exclude=./venv --exclude=./slurm_logs -cf - . | tar -C RiskAwareMCTS -xf -
cd RiskAwareMCTS

# Build a fresh Python $PYTHON_VERSION env on node-local scratch. Only the conda program itself is read from
# /cfs: the package cache and the env live under \$HOME (scratch), so the slow shared disk isn't loaded.
echo "\$(date '+%F %T')  building python $PYTHON_VERSION env with $CONDA_BIN"
export CONDA_PKGS_DIRS=\$HOME/conda_pkgs
$CONDA_BIN create -y -q -p \$HOME/env -c conda-forge --override-channels python=$PYTHON_VERSION
PY=\$HOME/env/bin/python
\$PY -m pip install -q -r requirements.txt
echo "\$(date '+%F %T')  env ready: \$(\$PY --version), \$(\$PY -c 'import numpy; print("numpy", numpy.__version__)')"

\$PY run_experiments.py --data-folder $DATA_DIR/ --num-processors \$SLURM_CPUS_PER_TASK $PY_ARGS
EOF
