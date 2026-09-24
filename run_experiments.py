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

Environments (SWEEP["envs"]): env name -> horizon H, or a dict {"H": ..., "gamma": ..., <env kwargs>}.
  - any name in envs.envs.MDPs (explicit MDPs; gamma comes from the MDP; all three algorithms available)
  - "binpacking": envs/binpacking_env_sequential.BinPackingEnv (gamma default 1, extra keys such as "num_bags" go to
    the constructor; no explicit MDP, so "erm-bi" is skipped for it)

Seeding: episode i of EVERY cell uses seed 1000 * base_seed + i (independent of env, algo, beta, n_iter), and both
np.random and Python's random are seeded inside the worker at the start of the episode. f_vals[i] therefore refers to
the same seed in every cell. Planning never consumes the real-step random streams (both are saved/restored around
every mcts.learn()), so all algorithms share the same initial state and, on the MDPs, the same random numbers for the
real transitions. On "binpacking" the item sequence is shared, but crush/spill draws (np.random inside step) only
happen on some branches, so those are only partly common across algorithms.

Note: K_ucb (sqrt(2)) and the ERM-MCTS best-action criterion ("min_erm") are fixed defaults here; they are not swept.
The MCTS episode loops below mirror simulate_erm_mcts.py / simulate_mcts_accrued_costs.py but work for any env.
"""

import argparse
import contextlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import pathlib
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime

import numpy as np

from algos.erm_backward_induction import ERMBackwardInduction
from algos.erm_mcts import ERMMCTS
from algos.mcts import MCTS
from envs.binpacking_env_sequential import BinPackingEnv
from envs.envs import get_env, MDPs
import simulate_mcts_accrued_costs as acc  # AccruedCosts_MDP, _safe_exp, NumpyEncoder

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent / "data") + "/"

ALGOS = ("erm-mcts", "acc-mcts", "erm-bi")

SWEEP = {
    "name": "demo",
    # env name -> horizon H, or {"H": ..., "gamma": ..., <env kwargs>} (see module docstring)
    "envs": {"four_state_mdp": 20, #ALWAYS USE H=20 FOR GRID-MDP
             #"two_paths_mdp": 15, #ALWAYS USE H=15 FOR GRID-MDP
             #"binpacking": {"H": 5, "gamma": 1, "num_bags": 5}
             },
    "algos": ["erm-mcts", "acc-mcts", "erm-bi"],
    "erm_betas": [0.001, 10, 20, 30, 50],
    "n_iters": [2000],  # n_iter_per_timestep (ignored by "erm-bi", which runs once per (env, beta))
    "N": 100,                     # episodes per cell
    "base_seed": 0,
    "num_processors": 8,
}


# --------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------

BINPACKING = "binpacking"


def env_kind(name):
    """"mdp" for explicit MDPs from envs.envs.MDPs, "binpacking" for BinPackingEnv."""
    return "binpacking" if name == BINPACKING else "mdp"


def parse_env_spec(env, spec):
    """SWEEP["envs"][env] (an int H, or a dict {"H", "gamma", <env kwargs>}) -> (H, gamma, env_kwargs)."""
    if env_kind(env) == "mdp" and env not in MDPs:
        raise ValueError(f"unknown env {env!r}; available: {list(MDPs) + [BINPACKING]}")
    if isinstance(spec, dict):
        kwargs = dict(spec)
        if "H" not in kwargs:
            raise ValueError(f"env {env!r}: the spec dict needs an 'H' (horizon) key")
        H, gamma = kwargs.pop("H"), kwargs.pop("gamma", None)
    else:
        H, gamma, kwargs = spec, None, {}

    if env_kind(env) == "mdp":
        if kwargs:
            raise ValueError(f"env {env!r} is an explicit MDP and takes no extra options: {sorted(kwargs)}")
        if gamma is not None and gamma != MDPs[env]["gamma"]:
            raise ValueError(f"env {env!r} has gamma={MDPs[env]['gamma']} built in; got gamma={gamma}")
        gamma = MDPs[env]["gamma"]
    elif gamma is None:
        gamma = 1  # BinPackingEnv default
    return H, gamma, kwargs


def cell_name(cell):
    n_iter = "NA" if cell["n_iter"] is None else cell["n_iter"]
    name = (f"{cell['env']}_{cell['algo']}_gamma_{cell['gamma']}_beta_{cell['erm_beta']}"
            f"_niter_{n_iter}_H_{cell['H']}")
    for key, value in sorted(cell["env_kwargs"].items()):  # e.g. _num_bags_5, so bag counts never collide
        name += f"_{key}_{value}"
    return name


def expand_cells(cfg):
    """Cartesian product env x algo x beta x n_iter. "erm-bi" ignores n_iter, so it gets a single
    cell (n_iter=None) per (env, beta) instead of identical re-runs, and only exists for explicit MDPs."""
    cells = []
    for env, spec in cfg["envs"].items():
        H, gamma, env_kwargs = parse_env_spec(env, spec)
        for algo, beta in itertools.product(cfg["algos"], cfg["erm_betas"]):
            if algo == "erm-bi" and env_kind(env) != "mdp":
                continue
            n_iters = [None] if algo == "erm-bi" else cfg["n_iters"]
            for n_iter in n_iters:
                cell = {"env": env, "H": H, "gamma": gamma, "env_kwargs": env_kwargs, "algo": algo,
                        "erm_beta": beta, "n_iter": n_iter}
                cell["name"] = cell_name(cell)
                cells.append(cell)
    return cells


def validate(cfg):
    for env, spec in cfg["envs"].items():
        parse_env_spec(env, spec)
    for algo in cfg["algos"]:
        if algo not in ALGOS:
            raise ValueError(f"unknown algo {algo!r}; available: {ALGOS}")


# --------------------------------------------------------------------------------------
# One episode (runs inside a worker process)
# --------------------------------------------------------------------------------------

def build_env(cell):
    """The raw (unwrapped) environment of a cell."""
    if env_kind(cell["env"]) == "binpacking":
        return BinPackingEnv(num_items_to_pack=cell["H"], gamma=cell["gamma"], **cell["env_kwargs"])
    return get_env(cell["env"], cell["H"])


class AccruedCosts_Env:
    """Accrued-cost wrapper for acc-mcts around any env that has no explicit MDP (e.g. BinPackingEnv).
    The state carries accrued_costs; the reward is 0 until termination, then exp(beta * accrued_costs)
    (see simulate_mcts_accrued_costs._safe_exp)."""

    def __init__(self, env, erm_beta):
        self.env = env
        self.gamma = env.gamma
        self.erm_beta = erm_beta

    def available_actions(self, state):
        return self.env.available_actions(state)

    def sample_initial_state(self):
        state = self.env.sample_initial_state()
        state["accrued_costs"] = 0.0
        return state

    def step(self, extended_state, a):
        next_state, cost, terminated = self.env.step(extended_state, a)
        next_accrued_cost = extended_state["accrued_costs"] + self.gamma ** extended_state["t"] * cost
        next_state["accrued_costs"] = next_accrued_cost
        if terminated:
            return next_state, acc._safe_exp(self.erm_beta * next_accrued_cost), True
        return next_state, 0.0, False


def build_accrued_env(cell):
    """The env acc-mcts plans in: accrued-cost augmented, terminal reward exp(beta * accrued cost)."""
    if env_kind(cell["env"]) == "binpacking":
        return AccruedCosts_Env(build_env(cell), cell["erm_beta"])
    return acc.AccruedCosts_MDP(MDPs[cell["env"]], cell["H"], cell["erm_beta"])


def _isolated_learn(mcts, n_iters):
    """mcts.learn() with both RNG streams restored afterwards, so planning never consumes the random
    numbers the real environment steps use (Python's random for items, np.random for the rest)."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    mcts.learn(n_iters=n_iters)
    random.setstate(py_state)
    np.random.set_state(np_state)


def simulate_erm_mcts(env, H, erm_beta, n_iter):
    """One ERM-MCTS episode; returns the discounted cumulative cost."""
    K_ucb = np.sqrt(2)

    def new_tree(state, root_depth):
        return ERMMCTS(initial_state=state, env=env, K_ucb=K_ucb, erm_beta=erm_beta,
                       rollout_policy=None, root_depth=root_depth)

    state = env.sample_initial_state()
    mcts = new_tree(state, 0)
    total = 0.0
    for t in range(H):
        _isolated_learn(mcts, n_iter)
        action = mcts.best_action()
        state, cost, terminated = env.step(state, action)
        total += cost * env.gamma ** t
        if terminated:
            break
        if mcts.update_root_node(action, state):
            mcts.set_root_depth(t + 1)
        else:
            mcts = new_tree(state, t + 1)  # next state not in the tree: rebuild it
    return total


def simulate_acc_mcts(env, H, erm_beta, n_iter):
    """One acc-mcts episode on the accrued-cost env; returns the discounted cumulative cost."""
    K_ucb = np.sqrt(2)

    def new_tree(state):
        return MCTS(initial_state=state, env=env, K_ucb=K_ucb, erm_beta=erm_beta, rollout_policy=None)

    state = env.sample_initial_state()
    mcts = new_tree(state)
    for t in range(H):
        _isolated_learn(mcts, n_iter)
        action = mcts.best_action()
        state, _, terminated = env.step(state, action)
        if terminated:
            break
        if not mcts.update_root_node(action, state):
            mcts = new_tree(state)
    return state["accrued_costs"]


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
    random.seed(seed)
    H, beta, n_iter = cell["H"], cell["erm_beta"], cell["n_iter"]

    # Keep any stray prints from the envs/algorithms out of the worker output.
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        if cell["algo"] == "erm-mcts":
            f_val = simulate_erm_mcts(build_env(cell), H, beta, n_iter)
        elif cell["algo"] == "acc-mcts":
            f_val = simulate_acc_mcts(build_accrued_env(cell), H, beta, n_iter)
        else:  # erm-bi
            f_val = _rollout_bi_policy(build_env(cell), H, bi_policy)

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
            bi_policy = ERMBackwardInduction(build_env(cell),
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


def main(cfg, data_folder_path=None, resume_dir=None, num_processors=None):
    data_folder_path = data_folder_path or DATA_FOLDER_PATH
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if resume_dir:
        sweep_dir = os.path.abspath(resume_dir)
        with open(os.path.join(sweep_dir, "sweep_config.json")) as f:
            sweep_meta = json.load(f)
        cfg = sweep_meta["sweep"]  # keep the original config so seeds/cells stay coherent
        if num_processors:  # a machine setting, not part of the experiment: may differ on resume
            cfg = {**cfg, "num_processors": num_processors}
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
    if "erm-bi" in cfg["algos"]:
        for env in cfg["envs"]:
            if env_kind(env) != "mdp":
                logger.info(f"erm-bi skipped for {env!r}: no explicit MDP (backward induction needs P and C)")

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
    parser.add_argument("--num-processors", type=int,
                        help="override SWEEP['num_processors'] (also applies with --resume)")
    args = parser.parse_args()

    cfg = dict(SWEEP)
    if args.base_seed is not None:
        cfg["base_seed"] = args.base_seed
    if args.name:
        cfg["name"] = args.name
    if args.num_processors:
        cfg["num_processors"] = args.num_processors
    main(cfg, data_folder_path=args.data_folder, resume_dir=args.resume, num_processors=args.num_processors)
