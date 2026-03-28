import torch
from algorithms import Gridworld, DynaQ, DynaQ_plus, DynaQ_plus_action_bonus, DynaQ_plus_selective_sample
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

gridsize = 6, 9
no_states = gridsize[0] * gridsize[1]
no_actions = 4
no_runs = 10
max_steps_envA = 3000
max_steps_envB = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(device)

env = Gridworld(device, no_runs, gridsize, goal_state=(0, 8))

def set_wall_a(env):
    env.reset_wall()
    env.wall[3, 1:9] = True

def set_wall_b(env):
    # enable reset if walls are added
    # env.reset() 
    env.reset_wall()
    env.wall[3, 1:8] = True

algorithms = {"DynaQ" : DynaQ, 
              "DynaQ+ Selective" : DynaQ_plus_selective_sample, 
              "DynaQ+" : DynaQ_plus, 
              "DynaQ+ Action Bonus" : DynaQ_plus_action_bonus}

def run_algorithms(algorithms, planning_steps,kappa=0.001, max_steps_envA= 5000,max_steps_envB= 5000):
    results = {}
    for name, algorithm in tqdm(algorithms.items(), leave=False, desc="Running Algorithms"):
        env.reset()
        agent = algorithm(device,no_runs, no_states, no_actions, env,kappa=kappa, gamma=0.8)
        set_wall_a(agent.env)
        left = agent.train(max_steps_envA, planning_steps)
        set_wall_b(agent.env)
        right = agent.train(max_steps_envB, planning_steps)

        results[name] = torch.concat((left,right + left[-1])).cpu()

    return results

def run_planning_study(start=1, stop=50, step= 1,kappa=0.001, algorithms= algorithms, max_steps_envA = max_steps_envA, max_steps_envB=max_steps_envB):
    data = {}
    for planning_steps in tqdm(range(start, stop, step), desc="Iter over planning steps"):
        results = run_algorithms(algorithms, planning_steps=planning_steps, kappa=kappa,
                                max_steps_envA=max_steps_envA, max_steps_envB=max_steps_envB)        
        data[planning_steps] = results
    return data

def run_kappa_study(start=1, stop=10, step=1,scale=1e-4,
                        planning_steps=50, algorithms=algorithms,
                        max_steps_envA=max_steps_envA, max_steps_envB=max_steps_envB):
    data = {}
    for kappa_key in tqdm(range(start, stop, step), desc="Iter over kappa"):
        kappa = kappa_key *  scale
        results = run_algorithms(algorithms, planning_steps=planning_steps, kappa=kappa,
                                max_steps_envA=max_steps_envA, max_steps_envB=max_steps_envB)        
        data[kappa_key] = results
    return data

def plot(results, filename=None, show= True):
    styles = [('default', 'light_imgs'), ('dark_background', 'dark_imgs')]
        
    for i, (style, folder) in enumerate(styles):
        plt.style.use(style)
        plt.figure(figsize=(12, 4.5))
        
        # Plotting logic
        for label, data in results.items():
            plt.plot(data, label=label)
        
        plt.xlabel("Steps", fontsize=18)
        plt.ylabel("Cumulative Reward", fontsize=18)
        plt.legend(fontsize=16)
        
        if filename is not None:
            save_path = f"content/posts/Reinforcement Learning/Planning and learning/programming_task/{folder}/{filename}.svg"
            plt.savefig(save_path, bbox_inches="tight", transparent=True)

        if i == len(styles) -1 and show:
            plt.show()

        plt.close() 

# data = run_planning_study(start=21, stop=51, max_steps_envA=6000,max_steps_envB = 6000)
data = run_kappa_study(start=1, stop=11, step=9,planning_steps= 1, max_steps_envA=3000,max_steps_envB = 3000)
# run_algorithms(algorithms,planning_steps=1, max_steps_envA=6000,max_steps_envB = 6000)
with open('content/posts/Reinforcement Learning/Planning and learning/programming_task/low_kappa_1e-4.pickle', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)