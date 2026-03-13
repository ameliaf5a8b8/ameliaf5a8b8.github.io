import matplotlib.pyplot as plt
from typing import Self, Optional, Dict, Any

def plot_and_save(
    data: list[tuple[str, str, list]], 
    xlabel: str = "Steps", 
    ylabel: str = "y", 
    filename: Optional[str] = None, 
    show: bool = True,
    figsize=(12,6)
):
    """
    Plots multiple datasets and optionally saves them in light and dark styles.

    Parameters:
    -----------
    data : list
        Contains tuples where the first values are labels, the next the colour of each plot, and the last lists of data points.
        If no color or no label is preffered, set the value of them to None
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    filename : str, optional
        Base filename to save the plots. Saved in both light and dark styles.
    show : bool
        Whether to display the plot.
    """

    def _plot(style: str):
        """Internal function to plot the data with a given style."""
        plt.figure(figsize=figsize)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if filename:
            plt.savefig(f"{filename}.svg", bbox_inches="tight", transparent=True)

            
    for label, colour, d in data:
        plot_kwargs = {}
        if label is not None:
            plot_kwargs['label'] = label
        if colour is not None:
            plot_kwargs['color'] = colour  # Note: Matplotlib uses 'color', not 'colour'
        
        plt.plot(d, **plot_kwargs)


    # Light style
    plt.style.use("default")
    _plot("light")

    # Dark style
    plt.style.use("dark_background")
    _plot("dark")


    # Show plot if requested
    if show:
        plt.show()








import numpy as np
import matplotlib.pyplot as plt



# plt.style.use("dark_background")

x = np.linspace(-3, 3, 400)

# x^2 function
y_quad = x**2

# Tangent point past minimum
x_tangent_point = 2
y_tangent_point = x_tangent_point**2
slope = 2 * x_tangent_point
y_tangent = slope * (x - x_tangent_point) + y_tangent_point

# Plotting
plt.figure(figsize=(8,6))
plt.plot(x, y_quad,  linewidth=2)
plt.plot(x, y_tangent,  linewidth=2)

# Labels
plt.xlabel('w', fontsize=18)
plt.ylabel('Loss', fontsize=18)

# Remove axis numbers
plt.xticks([])
plt.yticks([])

# Remove the black border (spines)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Center around the turning point (0,0)
plt.xlim(-3, 3)
plt.ylim(-1, 9)

# Adjust aspect ratio to make parabola look steeper
plt.gca().set_aspect(0.5)

plt.savefig(f"light.svg", bbox_inches="tight", transparent=True)
plt.show()