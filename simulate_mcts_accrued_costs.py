import os
import sys
import json
import multiprocessing as mp
import pathlib
from datetime import datetime

import numpy as np
from tqdm import tqdm

from algos.mcts import MCTS
from envs.envs import MDPs

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)
DEBUG = False

CONFIG = {
    "N": 2, # Number of experiments to run.
    "num_processors": 2,
    "env": "four_state_mdp",
    "H": 100, # Truncation length.
    "n_iter_per_timestep": 100, # MCTS number of tree expansion steps per timestep.
    "erm_beta": 0.1,
}

def create_exp_name(args: dict) -> str:
    return args['env'] + '_' + args['algo'] + '_gamma_' + str(args['gamma']) + '_beta_' + str(args['erm_beta']) + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S')) + str(args['seed'])


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _safe_exp(beta_times_cost):
    """np.exp, unclipped (experiment: overflow guard removed now that episode costs are
    normalized to ~[0, 1] for binpacking, so beta*cost is expected to stay far below the
    float64 overflow threshold ~709.78)."""
    return np.exp(beta_times_cost)


class AccruedCosts_MDP:

    def __init__(self, mdp, H, erm_beta):

        self.mdp = mdp
        self.H = H
        self.gamma = mdp["gamma"]
        self.erm_beta = erm_beta

    def available_actions(self, state):
        return self.mdp["actions"]

    def sample_initial_state(self):
        # Sample initial state.
        state = np.random.choice(self.mdp["states"], p=self.mdp["p_0"])
        extended_state = {"state": state,
                          "accrued_costs": 0.0,
                          "t": 0} # (state, accrued costs, timestep).
        return extended_state
    
    def step(self, extended_state, a):

        # Simulate step.
        state_t, accrued_costs_t, timestep_t = extended_state["state"], extended_state["accrued_costs"], extended_state["t"]
        next_accrued_cost = accrued_costs_t + self.gamma**timestep_t * self.mdp["C"][state_t, a]
        next_state = np.random.choice(self.mdp["states"], p=self.mdp["P"][a,state_t,:])
        next_timestep = timestep_t + 1
        next_extended_state = {"state": next_state,
                               "accrued_costs": next_accrued_cost,
                               "t": next_timestep}

        if DEBUG:
            print("state:", extended_state)
            print("action:", a)
            print("cost:", self.mdp["C"][state_t, a])

        if next_timestep == self.H:
            cost = _safe_exp(self.erm_beta * next_accrued_cost)
            terminated = True
        else:
            cost = 0.0
            terminated = False

        return next_extended_state, cost, terminated


    
def get_env(env_name, H, erm_beta):
    env_dict = MDPs[env_name]
    return AccruedCosts_MDP(env_dict, H, erm_beta)


def simulate_accrued_MCTS(env, H, erm_beta, n_iter_per_timestep=1_000):

    # Sample initial state.
    extended_state = env.sample_initial_state()

    K_ucb = np.sqrt(2)

    mcts = MCTS(initial_state=extended_state, env=env, K_ucb=K_ucb, erm_beta=erm_beta, rollout_policy=None)

    # Simulate until termination.
    cumulative_cost = 0.0
    for t in tqdm(range(H)):

        # Planning draws from the same np.random stream as the real env step; restore the
        # RNG state afterwards so the real transition depends only on prior real steps
        # (not on n_iter_per_timestep or on which tree shape the algorithm explored).
        np_state = np.random.get_state()
        mcts.learn(n_iters=n_iter_per_timestep)
        np.random.set_state(np_state)

        selected_action = mcts.best_action()

        # Environment step.
        cumulative_cost += env.mdp["C"][extended_state["state"], selected_action] * env.mdp["gamma"]**t
        extended_state, cost, terminated = env.step(extended_state, selected_action)

        updated_root = mcts.update_root_node(selected_action, extended_state)
        if not updated_root:
            # Next state is not present in the tree - build a new tree.
            mcts = MCTS(initial_state=extended_state, env=env, K_ucb=K_ucb, erm_beta=erm_beta, rollout_policy=None)

    print("final discounted cumulative cost:", cumulative_cost)

    return cumulative_cost


def run(cfg, seed):

    print('Running seed=', seed)

    np.random.seed(seed)

    # Instantiate environment.
    env = get_env(cfg["env"], cfg["H"], cfg["erm_beta"])

    mcts_f_val = simulate_accrued_MCTS(env=env,
                               H=cfg["H"],
                               erm_beta=cfg["erm_beta"],
                               n_iter_per_timestep=cfg["n_iter_per_timestep"])

    return mcts_f_val


def main(cfg, arg_seed=0, data_folder_path=None):

    if not data_folder_path:
        data_folder_path = DATA_FOLDER_PATH

    # Setup experiment data folder.
    exp_name = create_exp_name({'env': cfg['env'],
                                'algo': "acc-mcts",
                                'gamma': MDPs[cfg['env']]['gamma'],
                                'erm_beta': cfg['erm_beta'],
                                'seed': arg_seed,
                                })
    exp_path = data_folder_path + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('Config:')
    print(cfg)

    # Simulate.
    print('\nSimulating...')

    with mp.Pool(processes=cfg["num_processors"]) as pool:
        f_vals = pool.starmap(run, [(cfg, (1_000*arg_seed) + t) for t in range(cfg["N"])])
        pool.close()
        pool.join()

    f_vals = np.array(f_vals)

    exp_data = {}
    exp_data["config"] = cfg
    exp_data["f_vals"] = f_vals
    exp_data["env"] = cfg['env']

    # Dump dict.
    f = open(exp_path + "/exp_data.json", "w")
    dumped = json.dumps(exp_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print(exp_name)

    return exp_name


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(CONFIG, int(sys.argv[1]), sys.argv[2])
    else:
        main(CONFIG)