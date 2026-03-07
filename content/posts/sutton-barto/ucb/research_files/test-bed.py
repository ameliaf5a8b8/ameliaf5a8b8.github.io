import numpy as np
from tqdm import tqdm
import pickle

class Bandit:
    def __init__(self,Q_init, runs, steps, k, alpha):
        self.Q_init = Q_init
        self.q_true = np.random.normal(0, 1, (runs, k))
        self.Q = np.full((runs, k), self.Q_init, dtype=np.float32)

        self.optimal_arm = np.argmax(self.q_true, axis=1)
        self.optimal_action = np.zeros(steps)

        self.runs = runs
        self.steps = steps
        self.k = k
        self.alpha = alpha

        self.rewards = np.zeros(self.steps)
        
    def run_ucb_bandit(self,c):

        n_action_selected = np.zeros((self.runs,self.k))

        for t in tqdm(range(self.steps)):

            picked = n_action_selected != 0

            # The score of each action, assuming that we have already picked each action once
            # Thre will be a div by zero warning
            action_scores = self.Q + c * np.sqrt(np.log(t+1)/n_action_selected)

            # If we have not picked an option, we want to pick it next
            Actions = np.argmax(np.where(picked, action_scores, np.inf), axis=-1)

            n_action_selected[np.arange(self.runs), Actions] += 1


            rewards = np.random.normal(self.q_true[np.arange(self.runs), Actions], 1)

            self.rewards[t]  = rewards.mean()

            self.Q[np.arange(self.runs), Actions] += self.alpha * (
                rewards - self.Q[np.arange(self.runs), Actions]
            )

            self.optimal_action[t] = np.mean(Actions == self.optimal_arm)

        return self.optimal_action


    def run_epsilon_greedy_bandit(self,epsilon):

        for t in tqdm(range(self.steps)):

            explore = np.random.rand(self.runs) < epsilon

            greedy_actions = np.argmax(self.Q, axis=1)
            random_actions = np.random.randint(self.k, size=self.runs)

            actions = np.where(explore, random_actions, greedy_actions)

            rewards = np.random.normal(self.q_true[np.arange(self.runs), actions], 1)

            self.Q[np.arange(self.runs), actions] += self.alpha * (
                rewards -self.Q[np.arange(self.runs), actions]
            )
            
            self.rewards[t]  = rewards.mean()

            self.optimal_action[t] = np.mean(actions ==self.optimal_arm)

        return self.optimal_action

if __name__ == "__main__":
    runs = 5000
    steps = 1000
    k = 10
    alpha = 0.1
    c = 1
    epsilon = 0.1

    Simulation_ucb = Bandit(0, runs, steps, k, alpha)
    ucb = Simulation_ucb.run_ucb_bandit(c)

    Simulation_eps = Bandit(0, runs, steps, k, alpha)
    epsilon_greedy = Simulation_eps.run_epsilon_greedy_bandit(epsilon)

    with open('content/posts/sutton-barto/ucb/data/ucb_optimal_action.pkl', 'wb') as f:
        pickle.dump(ucb * 100, f)

    with open('content/posts/sutton-barto/ucb/data/eps_optimal_action.pkl', 'wb') as f:
        pickle.dump(epsilon_greedy * 100, f)

    with open('content/posts/sutton-barto/ucb/data/ucb_reward.pkl', 'wb') as f:
        pickle.dump(Simulation_ucb.rewards, f)

    with open('content/posts/sutton-barto/ucb/data/eps_reward.pkl', 'wb') as f:
        pickle.dump(Simulation_eps.rewards, f)


