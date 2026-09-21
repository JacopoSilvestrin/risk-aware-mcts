import numpy as np
from tqdm import tqdm

class DecisionNode:
    """
    The Decision Node class.

    A decision node stores a set of children nodes corresponding
    to the possible actions (of type RandomNode).

    :param state: (tuple) defining the state.
    :param available_actions: (tuple) defining the available actions at the state.
    :param father: (RandomNode) The father node; None if root.
    :param is_root: (bool)
    :param is_final: (bool)
    """

    def __init__(self, state=None, available_actions=None,
                    father=None, is_root=False, is_final=False):
        self.state = state
        self.available_actions = available_actions
        self.children = {}
        self.is_final = is_final
        self.visits = 0
        self.father = father
        self.is_root = is_root

        # Initialize children nodes.
        for action in self.available_actions:
            self.children[action] = RandomNode(action, father=self)

    def get_random_node(self, action: int):
        return self.children[action]

    def __repr__(self):
        s = ""
        for k, v in self.__dict__.items():
            if k == "children":
                print(k, v)
            elif k == "father":
                pass
            else:
                s += str(k)+": "+str(v)+"\n"
        return s


class RandomNode:
    """
    The RandomNode class defined by the state and the action.

    A random node stores a set of children nodes corresponding
    to the possible next states (of type DecisionNode).

    :param action: (int) taken in the decision node
    :param father: (DecisionNode)

    """
    def __init__(self, action: int, father: DecisionNode):
        self.action = action
        self.father = father
        self.children = {}
        self.cumulative_reward = 0
        self.visits = 0

    def add_child(self, child: DecisionNode, hash_preprocess):
        state_hash = hash_preprocess(child.state)
        if state_hash not in self.children.keys():
            self.children[state_hash] = child
        else:
            del child
            child = self.children[state_hash]

        return child

    def __repr__(self):
        if self.visits > 0:
            mean_rew = round(self.cumulative_reward/(self.visits), 2)
        else:
            mean_rew = np.nan
        return "[action: {}\nmean_reward: {}\nvisits: {}]".format(self.action, mean_rew, self.visits)

class MCTS:
    """
    Base class for MCTS based on Monte Carlo Tree Search for
    Continuous and Stochastic Sequential Decision Making Problems.

    :param initial_state: (int or tuple) initial state of the tree.
    :param env: environment function. Must implement:
            - step function that receives a state and an action and
                returns the next state, reward, and whether the env terminated.
            - available_actions that receives a state and returns a list with
                the available actions for that state.
    :param K_ucb: (float) exporation parameter of UCB
    :param erm_beta: (float) risk-sensitivity parameter used by the accrued-cost wrapper env
            to compute the terminal reward as -exp(erm_beta*accrued_cost). Needed here (not
            just in the env) to read that reward back out in cost units via _estimate_erm(),
            so the UCB objective stays commensurate with the exploration bonus. No gamma/
            depth-decay is needed: unlike ERM-MCTS's recursive Bellman/ERM decomposition,
            acc-mcts applies beta exactly once, to the total accrued cost, at termination.
    :param rollout_policy: (func) policy to perform rollouts.
            If None then rollout policy is random.

    Adapted from https://github.com/martinobdl/MCTS
    """
    def __init__(self, initial_state, env, K_ucb, erm_beta, rollout_policy=None):
        self.K_ucb = K_ucb
        self.env = env
        self.erm_beta = erm_beta
        self.rollout_policy = rollout_policy

        # Create tree.
        self.root = self.init_tree(initial_state)

    def _hash_state(self, state):
        node_repr = str(state["state"]) + "_" + \
                    str(state["accrued_costs"]) + "_" + \
                    str(state["t"])
        return hash(node_repr)

    def init_tree(self, initial_state):
        # Initialize tree.
        avail_actions = self.env.available_actions(initial_state)
        root = DecisionNode(state=initial_state,
                available_actions=avail_actions, is_root=True)
        return root

    def grow_tree(self):
        """
        Monte Carlo Tree Seach main loop: (i) Selection; (ii) Expansion;
            (iii) Simulation (rollout); and (iv) Backup.
        """
        decision_node = self.root
        decision_node.visits += 1 # Increment visits counter at root.

        # Selection and expansion.
        while (not decision_node.is_final) and decision_node.visits > 0:

            # Select action (and retrieve the random node associated with the action).
            a = self.select(decision_node)
            random_node = decision_node.get_random_node(a)

            # Generate the next decision node (next state).
            (new_decision_node, r) = self.select_outcome(random_node)

            # Store reward in random node (negate cost -> reward, consistent with rollout()).
            random_node.reward = -r

            # Add next decision node to random node children. Internally checks whether
            # the created next_decision_node (next state) already exists in tree; if it
            # exists, the existing node is returned instead.
            new_decision_node = random_node.add_child(new_decision_node, self._hash_state)

            decision_node = new_decision_node

        # Simulation (rollout).
        cumulative_reward = self.rollout(decision_node)

        # Backup.
        while not decision_node.is_root:
            decision_node.visits += 1
            random_node = decision_node.father
            cumulative_reward += random_node.reward
            random_node.cumulative_reward += cumulative_reward
            random_node.visits += 1
            decision_node = random_node.father

    def _estimate_erm(self, node: RandomNode):
        """
        Empirical ERM cost estimate (1/beta)*log(mean(exp(beta*cost))), read back out of
        this node's existing reward accumulator instead of a raw cost list. Valid because
        acc-mcts's reward is always -exp(erm_beta*cost) at termination and 0 at every
        intermediate step (see grow_tree()/rollout()'s sign convention), so
        cumulative_reward/visits == -mean(exp(erm_beta*cost)) for any visited node. Keeping
        this in cost units (rather than the raw exp-scale reward) is what lets the UCB
        exploration bonus actually compete with the objective instead of being permanently
        dwarfed by it.

        :param node: (RandomNode) node to estimate, must have node.visits > 0.
        :return: (float) empirical ERM cost estimate.
        """
        mean_reward = node.cumulative_reward / node.visits
        return (1.0 / self.erm_beta) * np.log(-mean_reward)

    def select(self, x: DecisionNode):
        """
        Selects the action to play from the current decision node.

        :param x: (DecisionNode) current decision node.
        :return: (int) action.
        """
        def scoring(k):
            child = x.children[k]
            if child.visits > 0:
                return -self._estimate_erm(child) + \
                    self.K_ucb*np.sqrt(np.sqrt(x.visits)/child.visits)
            else:
                return np.inf

        return max(x.children, key=scoring)

    def select_outcome(self, random_node: RandomNode):
        """
        Given a RandomNode, returns a new DecisionNode corresponding to the next state.

        :param: random_node: (RandomNode) the random node from which selects the next state.
        :return: (DecisionNode) the selected Decision Node.
        """
        new_state, r, done = self.env.step(random_node.father.state, random_node.action)
        avail_actions = self.env.available_actions(new_state)
        new_decision_node = DecisionNode(state=new_state, father=random_node,
                            available_actions=avail_actions, is_final=done)
        return new_decision_node, r

    def rollout(self, initial_node: DecisionNode):
        """
        Evaluates a DecisionNode by performing a rollout with a given policy.

        :param initial_node: (DecisionNone) initial node.
        :return: (float) the cumulative reward observed during the tree traversing.
        """
        R = 0
        done = initial_node.is_final
        s = initial_node.state
        while not done:
            if self.rollout_policy:
                a = self.rollout_policy(s)
            else:
                # Random policy.
                avail_actions = self.env.available_actions(s)
                a = np.random.choice(avail_actions)

            s, r, done = self.env.step(s,a)
            R += r

        return (-1.0) * R

    def learn(self, n_iters, progress_bar=False):
        """
        Expand the tree.

        :param: n_iters: (int) number of tree traversals.
        :param: progress_bar: (bool) whether to show a progress bar (tqdm).
        """
        if progress_bar:
            iterations = tqdm(range(n_iters))
        else:
            iterations = range(n_iters)
        for _ in iterations:
            self.grow_tree()

    def best_action(self):
        """
        Returns the root action with the lowest empirical ERM cost estimate (no exploration
        bonus -- this is the final commit, not tree search). Mirrors ERMMCTS's "min_erm"
        criterion: since this objective is tail-dominated, visit counts don't reliably track
        ERM ranking the way they would for an expected-value objective.

        :return: (int) the selected action.
        """
        children = list(self.root.children.values())
        erms = [self._estimate_erm(node) if node.visits > 0 else np.inf for node in children]

        return children[np.argmin(erms)].action
    
    def update_root_node(self, selected_action : int, new_state : dict):
        """
            Updates the root node of the planning tree.

            :param: (int) selected_action: previously selected action.
            :param: (dict) new_state: the new state of the environment.
        """
        next_state_hash = self._hash_state(new_state)
        if next_state_hash in self.root.children[selected_action].children:
            self.root = self.root.children[selected_action].children[next_state_hash]
            self.root.is_root = True
            return True
        else:
            return False

if __name__ == "__main__":
    pass