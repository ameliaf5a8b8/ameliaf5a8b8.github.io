import torch
from algorithms import Gridworld, DynaQ, DynaQ_plus, DynaQ_plus_action_bonus, DynaQ_plus_selective_sample
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp

gridsize = 6, 9
no_states = gridsize[0] * gridsize[1]
no_actions = 4
no_runs = 150
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

def run_algorithms(algorithms, planning_steps,kappa, max_steps_envA,max_steps_envB):
    results = {}
    for name, algorithm in algorithms.items():
        env.reset()
        agent = algorithm(device,no_runs, no_states, no_actions, env, kappa = kappa, gamma=0.8)
        set_wall_a(agent.env)
        left = agent.train(max_steps_envA, planning_steps)
        set_wall_b(agent.env)
        right = agent.train(max_steps_envB, planning_steps)

        results[name] = torch.concat((left,right + left[-1])).cpu()

    return results

def _run_planning_step(args):
    planning_steps, algorithms, kappa, max_steps_envA, max_steps_envB = args
    results = run_algorithms(algorithms, planning_steps, kappa, max_steps_envA, max_steps_envB)
    return planning_steps, results

def run_kappa_study(start=0.0001, stop=0.001, step=0.0001,
                        planning_steps=50, algorithms=algorithms,
                        max_steps_envA=max_steps_envA, max_steps_envB=max_steps_envB,
                        processes=None):
    kappa_values = []
    v = start
    while v <= stop:
        kappa_values.append(round(v, 10))
        v = round(v + step, 10)

    tasks = [
        (planning_steps, algorithms, kappa, max_steps_envA, max_steps_envB)
        for kappa in kappa_values
    ]

    data = {}
    with mp.Pool(processes=processes) as pool:
        for kappa, results in tqdm(
            pool.imap_unordered(_run_planning_step, tasks),
            total=len(tasks),
            desc="Iter over kappa values",
        ):
            data[kappa] = results

    return dict(sorted(data.items()))

def run_planning_study(start=1, stop=50, step= 1,kappa= 0.001, algorithms= algorithms, max_steps_envA = max_steps_envA, max_steps_envB=max_steps_envB, processes=None):
    planning_range = list(range(start, stop, step))
    tasks = [
        (planning_steps, algorithms, kappa, max_steps_envA, max_steps_envB)
        for planning_steps in planning_range
    ]
    data = {}

    with mp.Pool(processes=processes) as pool:
        for planning_steps, results in tqdm(
            pool.imap_unordered(_run_planning_step, tasks),
            total=len(tasks),
            desc="Iter over planning steps",
        ):
            data[planning_steps] = results

    data = dict(sorted(data.items()))
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

if __name__ == "__main__":
    # data = run_planning_study(start=1, stop=6, kappa= 0.0005, max_steps_envA=6000,max_steps_envB = 6000, processes=5)
    data = run_kappa_study(start=0.0001, stop=0.001, step=0.0001, planning_steps=1, max_steps_envA=6000,max_steps_envB = 6000, processes=5)
    with open('content/posts/Reinforcement Learning/Planning and learning/programming_task/low_kappa.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
