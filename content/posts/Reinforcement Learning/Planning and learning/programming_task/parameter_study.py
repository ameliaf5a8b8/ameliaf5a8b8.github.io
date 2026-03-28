import os
import pickle

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pickle

from algorithms_param_study import (
    DynaQ,
    DynaQ_plus,
    DynaQ_plus_action_bonus,
    DynaQ_plus_selective_sample,
    Gridworld,
    set_wall_a,
    set_wall_b,
)

os.chdir("content/posts/Reinforcement Learning/Planning and learning/programming_task/data")


gridsize = (6, 9)
no_states = gridsize[0] * gridsize[1]
no_actions = 4
no_runs = 150
max_steps_envA = 3000
max_steps_envB = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
BASE_SEED = 1234


algorithms = {
    "DynaQ": DynaQ,
    "DynaQ+ Selective": DynaQ_plus_selective_sample,
    "DynaQ+ Action Bonus": DynaQ_plus_action_bonus,
    "DynaQ+": DynaQ_plus,
}


def _set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_algorithms_batched(
    algorithms,
    param_name,
    param_values,
    planning_steps,
    kappa,
    max_steps_envA=5000,
    max_steps_envB=5000,
    seed=None,
):
    results = {}
    param_values = list(param_values)
    no_params = len(param_values)

    if no_params == 0:
        raise ValueError("param_values must contain at least one value")

    env = Gridworld(device, no_runs, no_params, gridsize, goal_state=(0, 8))

    if param_name == "planning_steps":
        batched_planning_steps = torch.tensor(param_values, device=device, dtype=torch.long)
        batched_kappa = kappa
        result_keys = [int(v) for v in param_values]
    elif param_name == "kappa":
        batched_planning_steps = planning_steps
        batched_kappa = torch.tensor(param_values, device=device, dtype=torch.float32)
        result_keys = list(param_values)
    else:
        raise ValueError("param_name must be either 'planning_steps' or 'kappa'")

    for algorithm_idx, (name, algorithm) in enumerate(
        tqdm(algorithms.items(), leave=False, desc="Running Algorithms")
    ):
        if seed is not None:
            _set_seed(seed + algorithm_idx)

        env.reset()
        agent = algorithm(
            device,
            no_runs,
            no_params,
            no_states,
            no_actions,
            env,
            kappa=batched_kappa,
            gamma=0.8,
        )

        set_wall_a(agent.env)
        left = agent.train(max_steps_envA, batched_planning_steps)
        left = left.cpu()
        set_wall_b(agent.env)
        right = agent.train(max_steps_envB, batched_planning_steps).cpu()

        full_curve = torch.cat((left, right + left[:, -1:].clone()), dim=1)
        results[name] = {
            key: full_curve[param_idx]
            for param_idx, key in enumerate(result_keys)
        }

    return results


def run_algorithms_grid(algorithms,kappa_keys,kappa_values,planning_values,max_steps_envA=5000,max_steps_envB=5000,seed=None,):
    results = {}
    planning_values = list(planning_values)
    kappa_values = list(kappa_values)

    if not planning_values:
        raise ValueError("planning_values must contain at least one value")
    if not kappa_values:
        raise ValueError("kappa_values must contain at least one value")

    combo_keys = []
    combo_kappas = []
    combo_planning = []
    for kappa_key, kappa_value in tqdm(
        list(zip(kappa_keys, kappa_values)),
        total=len(kappa_keys),
        leave=False,
        desc="Building parameter grid",
    ):
        for planning_steps in planning_values:
            combo_keys.append((kappa_key, planning_steps))
            combo_kappas.append(kappa_value)
            combo_planning.append(planning_steps)

    no_params = len(combo_keys)
    env = Gridworld(device, no_runs, no_params, gridsize, goal_state=(0, 8))
    batched_kappa = torch.tensor(combo_kappas, device=device, dtype=torch.float32)
    batched_planning_steps = torch.tensor(combo_planning, device=device, dtype=torch.long)

    for algorithm_idx, (name, algorithm) in enumerate(
        tqdm(algorithms.items(), leave=False, desc="Running Algorithms")
    ):
        if seed is not None:
            _set_seed(seed + algorithm_idx)

        env.reset()
        agent = algorithm(
            device,
            no_runs,
            no_params,
            no_states,
            no_actions,
            env,
            kappa=batched_kappa,
            gamma=0.8,
        )

        print(f"Q size: {agent.Q.element_size() * agent.Q.nelement() / 1e9:.2f} GB")
        print(f"model size: {agent.model.element_size() * agent.model.nelement() / 1e9:.2f} GB")

        set_wall_a(agent.env)
        left = agent.train(max_steps_envA, batched_planning_steps).cpu()
        set_wall_b(agent.env)
        right = agent.train(max_steps_envB, batched_planning_steps).cpu()

        results[name] = {
            combo_key: (
                float(left[param_idx, -1].item()),
                float(right[param_idx, -1].item()),
            )
            for param_idx, combo_key in enumerate(combo_keys)
        }

        # full_curve = torch.cat((left, right + left[:, -1:].clone()), dim=1)
        # print(f"Memory: {full_curve.numel() * full_curve.element_size() / (1024**2):.2f} MB")


        # results[name] = {
        #     combo_key: full_curve[param_idx]
        #     for param_idx, combo_key in enumerate(combo_keys)
        # }

    return results


def run_kappa_study(
    start=1,
    stop=10,
    step=1,
    scale=1e-4,
    kappa_values=None,
    planning_steps=50,
    algorithms=algorithms,
    max_steps_envA=max_steps_envA,
    max_steps_envB=max_steps_envB,
):
    if kappa_values is None:
        kappa_keys = list(range(start, stop, step))
        kappa_values = [kappa_key * scale for kappa_key in kappa_keys]
    else:
        if torch.is_tensor(kappa_values):
            kappa_values = kappa_values.tolist()
        else:
            kappa_values = list(kappa_values)
        kappa_keys = list(range(len(kappa_values)))

    batched_results = run_algorithms_batched(
        algorithms=algorithms,
        param_name="kappa",
        param_values=kappa_values,
        planning_steps=planning_steps,
        kappa=None,
        max_steps_envA=max_steps_envA,
        max_steps_envB=max_steps_envB,
        seed=BASE_SEED + planning_steps * 1000,
    )

    data = {}
    for kappa_key, kappa_value in tqdm(
        list(zip(kappa_keys, kappa_values)),
        total=len(kappa_keys),
        desc=f"Iter over kappa {planning_steps} planning steps",
    ):
        data[kappa_key] = {
            algorithm_name: per_param[kappa_value]
            for algorithm_name, per_param in batched_results.items()
        }

    return data


def run_kappa_planning_study(
    kappa_start=1,
    kappa_stop=101,
    kappa_step=1,
    scale=1e-5,
    kappa_values=None,
    planning_start=1,
    planning_stop=26,
    planning_step=1,
    algorithms=algorithms,
    max_steps_envA=max_steps_envA,
    max_steps_envB=max_steps_envB,
):
    if kappa_values is None:
        kappa_keys = list(range(kappa_start, kappa_stop, kappa_step))
        kappa_values = [kappa_key * scale for kappa_key in kappa_keys]
    else:
        if torch.is_tensor(kappa_values):
            kappa_values = kappa_values.tolist()
        else:
            kappa_values = list(kappa_values)
        kappa_keys = list(range(len(kappa_values)))
    planning_values = list(range(planning_start, planning_stop, planning_step))

    batched_results = run_algorithms_grid(
        algorithms=algorithms,
        kappa_keys=kappa_keys,
        kappa_values=kappa_values,
        planning_values=planning_values,
        max_steps_envA=max_steps_envA,
        max_steps_envB=max_steps_envB,
        seed=BASE_SEED,
    )

    data = {}
    for kappa_key in tqdm(kappa_keys, desc="Collecting kappa slices", leave=False):
        data[kappa_key] = {}
        for planning_steps in planning_values:
            data[kappa_key][planning_steps] = {
                algorithm_name: per_param[(kappa_key, planning_steps)]
                for algorithm_name, per_param in batched_results.items()
            }

    return data


def run_planning_study(
    start=1,
    stop=50,
    step=1,
    kappa=0.001,
    algorithms=algorithms,
    max_steps_envA=max_steps_envA,
    max_steps_envB=max_steps_envB,
):
    planning_values = list(range(start, stop, step))

    batched_results = run_algorithms_batched(
        algorithms=algorithms,
        param_name="planning_steps",
        param_values=planning_values,
        planning_steps=None,
        kappa=kappa,
        max_steps_envA=max_steps_envA,
        max_steps_envB=max_steps_envB,
        seed=BASE_SEED + start * 1000,
    )

    data = {}
    for planning_steps in tqdm(
        planning_values,
        total=len(planning_values),
        desc="Iter over planning steps",
    ):
        data[planning_steps] = {
            algorithm_name: per_param[planning_steps]
            for algorithm_name, per_param in batched_results.items()
        }

    return data


def plot(results, filename=None, show=True):
    styles = [("default", "light_imgs"), ("dark_background", "dark_imgs")]

    for i, (style, folder) in enumerate(styles):
        plt.style.use(style)
        plt.figure(figsize=(12, 4.5))

        for label, data in results.items():
            plt.plot(data, label=label)

        plt.xlabel("Steps", fontsize=18)
        plt.ylabel("Cumulative Reward", fontsize=18)
        plt.legend(fontsize=16)

        if filename is not None:
            save_path = f"{folder}/{filename}.svg"
            plt.savefig(save_path, bbox_inches="tight", transparent=True)

        if i == len(styles) - 1 and show:
            plt.show()

        plt.close()


if __name__ == "__main__":
    kappa_values = torch.logspace(-4, 0, 1000)
    data = run_kappa_planning_study(
        kappa_values=kappa_values,
        planning_start=1,
        planning_stop=26,
        planning_step=1,
        max_steps_envA=6000,
        max_steps_envB=6000,
    )

    pickled_data = pickle.dumps(data)
    size_mb = len(pickled_data) / (1024 * 1024)
    print(f"Size of data: {size_mb:.2f} MB")
    
    with open("kappa_planning_study_logspace_planning_1_to_25_6000.pickle", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
