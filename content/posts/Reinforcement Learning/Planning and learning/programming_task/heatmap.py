import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import os

os.chdir("content/posts/Reinforcement Learning/Planning and learning/programming_task/data")

def plot_heatmaps(pickle_path, kappa_values):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    kappa_keys = list(data.keys())
    planning_keys = list(data[kappa_keys[0]].keys())
    algorithm_names = list(data[kappa_keys[0]][planning_keys[0]].keys())

    n_kappa = len(kappa_keys)
    n_planning = len(planning_keys)
    n_algs = len(algorithm_names)

    # Build arrays: shape [n_algs, n_kappa, n_planning]
    left_arr = np.zeros((n_algs, n_kappa, n_planning))
    right_arr = np.zeros((n_algs, n_kappa, n_planning))
    cummulative_arr = np.zeros((n_algs, n_kappa, n_planning))


    for ki, kappa_key in enumerate(kappa_keys):
        for pi, planning_key in enumerate(planning_keys):
            for ai, alg_name in enumerate(algorithm_names):
                left_val, right_val = data[kappa_key][planning_key][alg_name]
                left_arr[ai, ki, pi] = left_val
                right_arr[ai, ki, pi] = right_val
                cummulative_arr[ai, ki, pi] = left_val + right_val

    kappa_floats = np.array(kappa_values.tolist()) if torch.is_tensor(kappa_values) else np.array(kappa_values)
    planning_floats = np.array(planning_keys)

    def make_figure(arr, title):
        fig, axes = plt.subplots(1, n_algs, figsize=(5 * n_algs, 5), constrained_layout=True)
        fig.suptitle(title, fontsize=16)

        vmin, vmax = arr.min(), arr.max()

        for ai, alg_name in enumerate(algorithm_names):
            ax = axes[ai]
            im = ax.imshow(
                arr[ai],
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            # y axis: kappa on log scale
            n_yticks = 6
            ytick_indices = np.linspace(0, n_kappa - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_indices)
            ax.set_yticklabels([f"{kappa_floats[i]:.1e}" for i in ytick_indices])

            # x axis: planning steps
            n_xticks = min(n_planning, 6)
            xtick_indices = np.linspace(0, n_planning - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_indices)
            ax.set_xticklabels([str(planning_floats[i]) for i in xtick_indices])

            ax.set_title(alg_name, fontsize=12)
            ax.set_xlabel("Planning steps")
            if ai == 0:
                ax.set_ylabel("Kappa")

            fig.colorbar(im, ax=ax, shrink=0.8)

        return fig

    fig_left = make_figure(left_arr, "Cumulative reward — env A (stationary)")
    fig_right = make_figure(right_arr, "Cumulative reward — env B (after wall change)")
    fig_cum = make_figure(cummulative_arr, "Cumulative reward")

    return fig_left, fig_right, fig_cum


if __name__ == "__main__":
    kappa_values = torch.logspace(-4, 0, 1000)
    fig_left, fig_right , fig_cum = plot_heatmaps(
        "kappa_planning_study_logspace_planning_1_to_25_6000.pickle",
        kappa_values,
    )
    # fig_left.savefig("light_imgs/heatmap_envA.svg", bbox_inches="tight", transparent=True)
    # fig_right.savefig("light_imgs/heatmap_envB.svg", bbox_inches="tight", transparent=True)
    plt.show()