# Entropic Risk-Aware Monte Carlo Tree Search

Paper: https://www.arxiv.org/abs/2601.17667

Dependencies: numpy, tqdm, matplotlib, seaborn.

Tested with Python 3.8.10

## Running a sweep

`run_experiments.py` sweeps environments x algorithms (`erm-mcts`, `acc-mcts`, `erm-bi`) x `erm_beta` x
`n_iter_per_timestep`. Edit the `SWEEP` dict at the top of the file, then:

```
python run_experiments.py                          # run the sweep
python run_experiments.py --base-seed 3            # another shared set of episode seeds
python run_experiments.py --resume data/sweep_...  # finish an interrupted sweep (its saved config is used)
```

**Environments.** `SWEEP["envs"]` maps an env name to its horizon `H`, or to a dict `{"H": ..., "gamma": ..., ...}`:

```python
"envs": {"four_state_mdp": 20,                                         # any name in envs.envs.MDPs
         "binpacking": {"H": 5, "gamma": 1, "num_bags": 5}},           # envs/binpacking_env_sequential.py
```

Extra keys (such as `num_bags`) are passed to the `BinPackingEnv` constructor and appended to the cell folder name
(`..._H_5_num_bags_5`). MDPs have their own `gamma` built in and take no extra keys. `erm-bi` (exact backward
induction) needs explicit transition/cost tables, so it is only run on the MDPs and skipped, with a log line, for
`binpacking`.

**Seeding.** Episode `i` of every cell uses seed `1000 * base_seed + i`, whatever the env, algorithm, beta or
`n_iter`. So `f_vals[i]` is the same seeded episode in every cell, and cells can be compared per seed. Both
`np.random` and Python's `random` are seeded, and both are restored after each planning phase, so planning never
consumes the random numbers of the real transitions. On the MDPs the initial state and all real transitions'
random numbers are shared by every algorithm. On `binpacking` the initial state and the item sequence are shared,
but the crush/spill draws only happen on some branches, so those are only partly shared between algorithms.
`erm-bi` does not use `n_iter`, so it runs once per (env, beta).

**Output.** One folder per cell under `data/sweep_<name>_<timestamp>_seed<base_seed>/`:

| File | Content |
|---|---|
| `sweep_config.json` | the `SWEEP` used, git commit, start/resume times |
| `manifest.json` | every cell with status (`pending`/`done`/`failed`), seconds, mean, std, empirical ERM |
| `sweep.log` | sweep-level log (also printed to stdout) |
| `<env>_<algo>_gamma_<g>_beta_<b>_niter_<n>_H_<H>/exp_data.json` | `config`, `f_vals`, `seeds`, ... in the same format as the `simulate_*` scripts, so `merge_exps.py` and the notebooks work unchanged |
| `<cell>/run.log` | one line per episode: seed, f_val, seconds |
| `<cell>/decisions.jsonl` | `erm-mcts` only: one JSON line per episode with, at every step, the executed, most-visited and min-ERM root actions, whether the last two agree, and each root action's visits and empirical ERM. `manifest.json` also gets the disagreement count and rate |

## Running a sweep on Slurm

`run_sweep_slurm.sh` submits `run_experiments.py` as one Slurm job. Run it from the repo checkout on the cluster
after editing `SWEEP`:

```
./run_sweep_slurm.sh                                         # 50 CPUs, partition gaips_cpu_medium, 96h
./run_sweep_slurm.sh --cpus 32 --time 48:00:00 --mem 64G --base-seed 3 --name highbeta
./run_sweep_slurm.sh --resume data/sweep_demo_...            # finish a sweep that hit the time limit
```

The job copies the repo to `/scratch/slurm-jobs/$SLURM_JOB_ID` and builds a fresh Python 3.8 env there with conda
(`--python` and `--conda` to change the version or the conda binary; the default conda is `$CONDA_EXE` or
`~/miniconda3/bin/conda`), then installs `requirements.txt`. Conda's package cache and the env stay on the node's
scratch disk, so nothing is installed on `/cfs` and no conda env needs to exist or be activated before submitting.
Setup takes under a minute. The sweep runs with `--num-processors $SLURM_CPUS_PER_TASK`, so the pool size always
matches the CPUs requested. Results go straight to `--data-dir` (default `<repo>/data`) on the shared filesystem, and
the job's stdout (including the env setup) goes to `slurm_logs/sweep-<jobid>.out`.
