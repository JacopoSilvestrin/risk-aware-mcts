import os
import sys
import json
import multiprocessing as mp
import pathlib
from datetime import datetime

import numpy as np
from tqdm import tqdm

from algos.erm_mcts import ERMMCTS
from envs.envs import get_env, MDPs

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)
DEBUG = False

CONFIG = {
    "N": 2, # Number of experiments to run.
    "num_processors": 2,
    "env": "four_state_mdp",
    "H": 100, # Truncation length.
    "n_iter_per_timestep": 500, # MCTS number of tree expansion steps per timestep.
    "erm_beta": 0.5,
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

def simulate_ERM_MCTS(env, H, erm_beta, n_iter_per_timestep=1_000):

    # Sample initial state.
    extended_state = env.sample_initial_state()
    K_ucb = np.sqrt(2)

    mcts = ERMMCTS(initial_state=extended_state, env=env, K_ucb=K_ucb,
                    erm_beta=erm_beta, rollout_policy=None, root_depth=0)

    # Simulate until termination.
    cumulative_discounted_cost = 0.0
    for t in tqdm(range(H)):

        if DEBUG:
            print("state:", extended_state)

        # Planning draws from the same np.random stream as the real env step; restore the
        # RNG state afterwards so the real transition depends only on prior real steps
        # (not on n_iter_per_timestep or on which tree shape the algorithm explored).
        np_state = np.random.get_state()
        mcts.learn(n_iters=n_iter_per_timestep)
        np.random.set_state(np_state)

        selected_action = mcts.best_action()

        if DEBUG:
            print("action:", selected_action)

        # Environment step.
        extended_state, cost, terminated = env.step(extended_state, selected_action)
        cumulative_discounted_cost += cost * env.mdp["gamma"]**t

        if DEBUG:
            print("cost:", cost)

        updated_root = mcts.update_root_node(selected_action, extended_state)
        if updated_root:
            # Next state is present in the tree - reuse subtree.
            mcts.set_root_depth(t+1)
        else:
            # Next state is not present in the tree - build a new tree.
            mcts = ERMMCTS(initial_state=extended_state, env=env, K_ucb=K_ucb,
                    erm_beta=erm_beta, rollout_policy=None, root_depth=t+1)

    print("final discounted cumulative cost:", cumulative_discounted_cost)

    return cumulative_discounted_cost


def run(cfg, seed):

    print('Running seed=', seed)

    np.random.seed(seed)

    # Instantiate environment.
    env = get_env(cfg["env"], cfg["H"])

    mcts_f_val = simulate_ERM_MCTS(env=env,
                               H=cfg["H"],
                               erm_beta=cfg["erm_beta"],
                               n_iter_per_timestep=cfg["n_iter_per_timestep"])

    return mcts_f_val


def main(cfg, arg_seed=0, data_folder_path=None):

    if not data_folder_path:
        data_folder_path = DATA_FOLDER_PATH

    # Setup experiment data folder.
    exp_name = create_exp_name({'env': cfg['env'],
                                'algo': "erm-mcts",
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
