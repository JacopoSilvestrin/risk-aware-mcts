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

**Seeding.** Episode `i` of every cell uses seed `1000 * base_seed + i`, whatever the env, algorithm, beta or
`n_iter`. So `f_vals[i]` is the same seeded episode in every cell (same initial state, same random numbers for the
real transitions), and cells can be compared per seed. `erm-bi` (exact backward induction) does not use `n_iter`, so
it runs once per (env, beta).

**Output.** One folder per cell under `data/sweep_<name>_<timestamp>_seed<base_seed>/`:

| File | Content |
|---|---|
| `sweep_config.json` | the `SWEEP` used, git commit, start/resume times |
| `manifest.json` | every cell with status (`pending`/`done`/`failed`), seconds, mean, std, empirical ERM |
| `sweep.log` | sweep-level log (also printed to stdout) |
| `<env>_<algo>_gamma_<g>_beta_<b>_niter_<n>_H_<H>/exp_data.json` | `config`, `f_vals`, `seeds`, ... in the same format as the `simulate_*` scripts, so `merge_exps.py` and the notebooks work unchanged |
| `<cell>/run.log` | one line per episode: seed, f_val, seconds |
