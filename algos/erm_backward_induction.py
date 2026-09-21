import numpy as np

class ERMBackwardInduction(object):

    def __init__(self, env, beta, horizon):
        self.mdp = env.mdp
        self.gamma = env.gamma
        self.beta = beta
        self.horizon = horizon

    def compute(self):

        print('Running ERMBackwardInduction...')
        V_func = np.zeros((self.horizon, len(self.mdp["states"])))
        V_func[-1,:] = np.min(self.mdp["C"], axis=1)

        policy = np.zeros((self.horizon, len(self.mdp["states"])),dtype=np.int32)
        policy[-1,:] = np.argmin(self.mdp["C"], axis=1)
        
        for t in range(self.horizon-2,-1,-1):
            beta_t = self.beta * self.gamma**t

            for state in self.mdp["states"]:

                Q_vals = []
                for action in self.mdp["actions"]:
                    
                    x = beta_t * (self.mdp["C"][state][action] + self.gamma * V_func[t+1,:])
                    max_val = np.max(x)
                    exp_next_states = np.dot(self.mdp["P"][action][state], np.exp(x - max_val))
                    Q_vals.append((1.0/beta_t) * (max_val + np.log(exp_next_states)))

                V_func[t,state] = np.min(Q_vals)
                policy[t,state] = np.argmin(Q_vals)

        print("V_func", V_func)
        print("policy", policy)
        return policy
