"""
Sweep runner: environments x algorithms x erm_betas x n_iter_per_timestep, all evaluated on the
SAME seeded episodes so that results are paired and comparable across cells.

Usage:
    python run_experiments.py                          # run the SWEEP defined below
    python run_experiments.py --base-seed 3            # different (but still shared) set of episode seeds
    python run_experiments.py --resume data/sweep_...  # finish an interrupted sweep (uses its saved config)

Output layout (one folder per (env, algo, beta, n_iter) "cell"):

    data/sweep_<name>_<timestamp>_seed<base_seed>/
        sweep_config.json    exact SWEEP used, git commit, start / resume times
        manifest.json        one entry per cell: params, status (pending|done|failed), seconds, mean, std, erm
        sweep.log            sweep-level log (also echoed to stdout)
        <cell_name>/
            exp_data.json    {config, f_vals, seeds, ...} -- same double-encoded format as the simulate_* scripts,
                             so merge_exps.py and the notebooks (json.loads(json.load(f))["f_vals"]) keep working
            run.log          one line per episode: seed, f_val, seconds

Seeding: episode i of EVERY cell uses seed 1000 * base_seed + i (independent of env, algo, beta, n_iter), and
np.random.seed(seed) is called inside the worker at the start of the episode. f_vals[i] therefore refers to the
same seed in every cell. Planning never consumes the real-step random stream (see the RNG isolation in
simulate_erm_mcts.py / simulate_mcts_accrued_costs.py), so all algorithms also share the same initial state and the
same random numbers for the real transitions.

Note: K_ucb (sqrt(2)) and the ERM-MCTS best-action criterion ("min_erm") are the defaults hard-coded in the
simulate_* scripts this runner reuses; they are not swept here.
"""

import argparse
import contextlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import pathlib
import subprocess
import sys
import time
import traceback
from datetime import datetime

import numpy as np

from algos.erm_backward_induction import ERMBackwardInduction
from envs.envs import get_env, MDPs
from simulate_erm_mcts import simulate_ERM_MCTS
import simulate_mcts_accrued_costs as acc

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent / "data") + "/"

ALGOS = ("erm-mcts", "acc-mcts", "erm-bi")

SWEEP = {
    "name": "demo",
    "envs": {"four_state_mdp": 20, "two_paths_mdp": 15},  # env name -> horizon H
    "algos": ["erm-mcts", "acc-mcts", "erm-bi"],
    "erm_betas": [0.1, 0.5, 1.0],
    "n_iters": [100, 500, 2000],  # n_iter_per_timestep (ignored by "erm-bi", which runs once per (env, beta))
    "N": 100,                     # episodes per cell
    "base_seed": 0,
    "num_processors": 8,
}


# --------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------

def cell_name(cell):
    n_iter = "NA" if cell["n_iter"] is None else cell["n_iter"]
    return (f"{cell['env']}_{cell['algo']}_gamma_{cell['gamma']}_beta_{cell['erm_beta']}"
            f"_niter_{n_iter}_H_{cell['H']}")


def expand_cells(cfg):
    """Cartesian product env x algo x beta x n_iter. "erm-bi" ignores n_iter, so it gets a single
    cell (n_iter=None) per (env, beta) instead of identical re-runs."""
    cells = []
    for env, H in cfg["envs"].items():
        gamma = MDPs[env]["gamma"]
        for algo, beta in itertools.product(cfg["algos"], cfg["erm_betas"]):
            n_iters = [None] if algo == "erm-bi" else cfg["n_iters"]
            for n_iter in n_iters:
                cell = {"env": env, "H": H, "gamma": gamma, "algo": algo,
                        "erm_beta": beta, "n_iter": n_iter}
                cell["name"] = cell_name(cell)
                cells.append(cell)
    return cells


def validate(cfg):
    for env in cfg["envs"]:
        if env not in MDPs:
            raise ValueError(f"unknown env {env!r}; available: {list(MDPs)}")
    for algo in cfg["algos"]:
        if algo not in ALGOS:
            raise ValueError(f"unknown algo {algo!r}; available: {ALGOS}")


# --------------------------------------------------------------------------------------
# One episode (runs inside a worker process)
# --------------------------------------------------------------------------------------

def _rollout_bi_policy(env, H, policy):
    """Roll out a precomputed backward-induction policy (same loop/discounting as the simulate_* scripts)."""
    state = env.sample_initial_state()
    total = 0.0
    for t in range(H):
        action = int(policy[t, state["state"]])
        state, cost, _ = env.step(state, action)
        total += cost * env.gamma ** t
    return total


def run_episode(args):
    cell, seed, bi_policy = args
    t0 = time.time()
    np.random.seed(seed)
    env_name, H, beta, n_iter = cell["env"], cell["H"], cell["erm_beta"], cell["n_iter"]

    # The reused simulate_* functions print and draw tqdm bars; keep worker output out of the logs.
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        if cell["algo"] == "erm-mcts":
            f_val = simulate_ERM_MCTS(env=get_env(env_name, H), H=H, erm_beta=beta,
                                      n_iter_per_timestep=n_iter)
        elif cell["algo"] == "acc-mcts":
            f_val = acc.simulate_accrued_MCTS(env=acc.get_env(env_name, H, beta), H=H, erm_beta=beta,
                                              n_iter_per_timestep=n_iter)
        else:  # erm-bi
            f_val = _rollout_bi_policy(get_env(env_name, H), H, bi_policy)

    return seed, float(f_val), time.time() - t0


# --------------------------------------------------------------------------------------
# Bookkeeping helpers
# --------------------------------------------------------------------------------------

def empirical_erm(f_vals, beta):
    """(1/beta) * log(mean(exp(beta * f))), log-sum-exp stabilised."""
    x = beta * np.asarray(f_vals, dtype=float)
    m = np.max(x)
    return float((m + np.log(np.mean(np.exp(x - m)))) / beta)


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, cls=acc.NumpyEncoder)
    os.replace(tmp, path)


def _git_info():
    try:
        root = str(pathlib.Path(__file__).parent)
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                                             stderr=subprocess.DEVNULL).decode().strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _make_logger(path):
    logger = logging.getLogger("sweep:" + path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


# --------------------------------------------------------------------------------------
# One cell / whole sweep
# --------------------------------------------------------------------------------------

def run_cell(pool, cell, cell_dir, seeds):
    os.makedirs(cell_dir, exist_ok=True)

    bi_policy = None
    if cell["algo"] == "erm-bi":
        # Computed once per cell (the policy does not depend on the episode seed).
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            bi_policy = ERMBackwardInduction(get_env(cell["env"], cell["H"]),
                                             cell["erm_beta"], cell["H"]).compute()

    t0 = time.time()
    f_vals = []
    with open(os.path.join(cell_dir, "run.log"), "w") as run_log:
        # imap is ordered: f_vals[i] always corresponds to seeds[i].
        for seed, f_val, secs in pool.imap(run_episode, [(cell, s, bi_policy) for s in seeds]):
            f_vals.append(f_val)
            run_log.write(f"seed={seed} f_val={f_val:.10g} seconds={secs:.3f}\n")
            run_log.flush()

    exp_data = {
        "config": cell,
        "f_vals": np.array(f_vals),
        "seeds": list(seeds),
        "env": cell["env"],
        "algo": cell["algo"],
        "erm_beta": cell["erm_beta"],
        "n_iter_per_timestep": cell["n_iter"],
        "H": cell["H"],
    }
    # Same double encoding as the simulate_* scripts (json.loads(json.load(f)) reads it back).
    with open(os.path.join(cell_dir, "exp_data.json"), "w") as f:
        json.dump(json.dumps(exp_data, cls=acc.NumpyEncoder), f)

    return {"seconds": time.time() - t0,
            "mean": float(np.mean(f_vals)),
            "std": float(np.std(f_vals)),
            "erm": empirical_erm(f_vals, cell["erm_beta"])}


def _print_table(logger, manifest):
    rows = [c for c in manifest["cells"].values() if c["status"] == "done"]
    rows.sort(key=lambda c: (c["env"], c["algo"], c["erm_beta"], c["n_iter"] or 0))
    logger.info("")
    logger.info(f"{'env':<18}{'algo':<10}{'beta':>8}{'n_iter':>8}{'mean':>12}{'std':>10}{'ERM(f)':>12}")
    for c in rows:
        n_iter = "NA" if c["n_iter"] is None else c["n_iter"]
        logger.info(f"{c['env']:<18}{c['algo']:<10}{c['erm_beta']:>8g}{str(n_iter):>8}"
                    f"{c['mean']:>12.4f}{c['std']:>10.4f}{c['erm']:>12.4f}")


def main(cfg, data_folder_path=None, resume_dir=None):
    data_folder_path = data_folder_path or DATA_FOLDER_PATH
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if resume_dir:
        sweep_dir = os.path.abspath(resume_dir)
        with open(os.path.join(sweep_dir, "sweep_config.json")) as f:
            sweep_meta = json.load(f)
        cfg = sweep_meta["sweep"]  # keep the original config so seeds/cells stay coherent
        sweep_meta.setdefault("resumed_at", []).append(now)
    else:
        sweep_dir = os.path.join(data_folder_path, f"sweep_{cfg['name']}_{now}_seed{cfg['base_seed']}")
        sweep_meta = {"sweep": cfg, "started_at": now, "git": _git_info()}
    validate(cfg)
    os.makedirs(sweep_dir, exist_ok=True)
    _write_json_atomic(os.path.join(sweep_dir, "sweep_config.json"), sweep_meta)

    logger = _make_logger(os.path.join(sweep_dir, "sweep.log"))
    seeds = [1000 * cfg["base_seed"] + i for i in range(cfg["N"])]
    cells = expand_cells(cfg)

    manifest_path = os.path.join(sweep_dir, "manifest.json")
    if resume_dir and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"sweep_dir": sweep_dir, "seeds": seeds, "cells": {}}
    for cell in cells:
        manifest["cells"].setdefault(cell["name"], {**cell, "status": "pending"})
    _write_json_atomic(manifest_path, manifest)

    logger.info(f"sweep dir: {sweep_dir}")
    logger.info(f"{len(cells)} cells x {cfg['N']} episodes = {len(cells) * cfg['N']} episodes; "
                f"seeds {seeds[0]}..{seeds[-1]} shared by every cell; {cfg['num_processors']} processes")

    with mp.Pool(processes=cfg["num_processors"]) as pool:
        for k, cell in enumerate(cells, 1):
            entry = manifest["cells"][cell["name"]]
            cell_dir = os.path.join(sweep_dir, cell["name"])
            if entry["status"] == "done" and os.path.exists(os.path.join(cell_dir, "exp_data.json")):
                logger.info(f"[{k}/{len(cells)}] skip (done): {cell['name']}")
                continue

            logger.info(f"[{k}/{len(cells)}] start: {cell['name']}")
            try:
                stats = run_cell(pool, cell, cell_dir, seeds)
                entry.update(stats, status="done")
                entry.pop("error", None)
                logger.info(f"[{k}/{len(cells)}] done in {stats['seconds']:.1f}s: mean={stats['mean']:.4f} "
                            f"std={stats['std']:.4f} erm={stats['erm']:.4f}")
            except Exception as e:
                entry.update(status="failed", error=f"{type(e).__name__}: {e}")
                logger.info(f"[{k}/{len(cells)}] FAILED: {cell['name']}\n{traceback.format_exc()}")
            _write_json_atomic(manifest_path, manifest)

    n_failed = sum(c["status"] == "failed" for c in manifest["cells"].values())
    _print_table(logger, manifest)
    logger.info(f"finished: {len(cells) - n_failed}/{len(cells)} cells ok, {n_failed} failed")
    return sweep_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-seed", type=int, help="override SWEEP['base_seed']")
    parser.add_argument("--name", help="override SWEEP['name']")
    parser.add_argument("--data-folder", help="where to create the sweep folder (default: ./data/)")
    parser.add_argument("--resume", metavar="SWEEP_DIR",
                        help="continue an interrupted sweep: finished cells are skipped and the SAVED config is used")
    args = parser.parse_args()

    cfg = dict(SWEEP)
    if args.base_seed is not None:
        cfg["base_seed"] = args.base_seed
    if args.name:
        cfg["name"] = args.name
    main(cfg, data_folder_path=args.data_folder, resume_dir=args.resume)
