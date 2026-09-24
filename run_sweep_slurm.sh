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

set -e

REPO_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_DIR=$REPO_DIR/data
PARTITION=gaips_phd
CPUS=50
TIME=96:00:00
MEM=
PY_ARGS=

while [ $# -gt 0 ]; do
    case $1 in
    --partition) PARTITION=$2 ; shift ;;
    --cpus) CPUS=$2 ; shift ;;
    --time) TIME=$2 ; shift ;;
    --mem) MEM=$2 ; shift ;;              # e.g. 64G; default: the partition's default
    --data-dir) DATA_DIR=$(realpath "$2") ; shift ;;
    --base-seed) PY_ARGS="$PY_ARGS --base-seed $2" ; shift ;;
    --name) PY_ARGS="$PY_ARGS --name $2" ; shift ;;
    --resume) PY_ARGS="$PY_ARGS --resume $(realpath "$2")" ; shift ;;
    -h|--help) sed -n '2,12p' "$0" ; exit 0 ;;
    *) echo "$0: unrecognized option $1" 1>&2 ; exit 1 ;;
    esac
    shift
done

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

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python run_experiments.py --data-folder $DATA_DIR/ --num-processors \$SLURM_CPUS_PER_TASK $PY_ARGS
EOF
