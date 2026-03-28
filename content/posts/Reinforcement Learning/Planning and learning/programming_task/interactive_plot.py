import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
import os

os.chdir("content/posts/Reinforcement Learning/Planning and learning/programming_task")


def interactive_plot(data):
    planning_steps = np.array(sorted(data))
    if planning_steps.size == 0:
        raise ValueError("data must contain at least one planning step")

    algorithm_names = sorted({name for step in data.values() for name in step})
    first_step = data[int(planning_steps[0])]

    def get_series(run, name):
        values = run.get(name)
        if values is None:
            return np.array([], dtype=float)
        return np.asarray(values, dtype=float).reshape(-1)

    def get_bounds(run):
        lengths = []
        mins = []
        maxs = []

        for name in algorithm_names:
            series = get_series(run, name)
            if series.size == 0:
                continue
            lengths.append(series.size)
            mins.append(float(series.min()))
            maxs.append(float(series.max()))

        if not lengths:
            return 1, -1.0, 1.0

        y_min = min(mins)
        y_max = max(maxs)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        return max(lengths), y_min, y_max

    max_len, y_min, y_max = get_bounds(first_step)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.22)

    lines = {}

    for name in algorithm_names:
        series = get_series(first_step, name)
        line, = ax.plot(np.arange(series.size), series, label=name, linewidth=2)
        line.set_visible(series.size > 0)
        lines[name] = line

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"Planning Steps: {int(planning_steps[0])}")
    ax.set_xlim(0, max_len - 1 if max_len > 1 else 1)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    slider_ax = fig.add_axes((0.12, 0.08, 0.76, 0.04))
    slider = Slider(
        ax=slider_ax,
        label="Planning Steps",
        valmin=0,
        valmax=len(planning_steps) - 1,
        valinit=0,
        valstep=1,
    )

    def update(_):
        step = int(planning_steps[int(slider.val)])
        run = data[step]
        max_len, local_min, local_max = get_bounds(run)

        for name in algorithm_names:
            series = get_series(run, name)
            lines[name].set_data(np.arange(series.size), series)
            lines[name].set_visible(series.size > 0)

        ax.set_xlim(0, max_len - 1 if max_len > 1 else 1)
        ax.set_ylim(local_min, local_max)
        ax.set_title(f"Planning Steps: {step}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


with open("low_kappa.pickle", "rb") as f:
    data = pickle.load(f)
plt.style.use("dark_background")
interactive_plot(data)
