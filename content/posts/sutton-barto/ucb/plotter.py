import matplotlib.pyplot as plt
import pickle

with open('content/posts/sutton-barto/ucb/data/ucb_optimal_action.pkl', 'rb') as f:
    ucb_optimal_action = pickle.load(f)

with open('content/posts/sutton-barto/ucb/data/eps_optimal_action.pkl', 'rb') as f:
    eps_optimal_action = pickle.load(f)

with open('content/posts/sutton-barto/ucb/data/ucb_reward.pkl', 'rb') as f:
    ucb_reward = pickle.load(f)

with open('content/posts/sutton-barto/ucb/data/eps_reward.pkl', 'rb') as f:
    eps_reward = pickle.load(f)

print(ucb_reward[-10:])
print(eps_reward[-10:])
# plt.style.use('dark_background')


def plot_and_save(data1, data2, data1_label, data2_label, xlabel = "Steps", ylabel = "y", filename=None):
    plt.figure(figsize=(12,4.5))

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    plt.plot(data1, label= data1_label)
    plt.plot(data2, label=data2_label)
    plt.legend(fontsize=16)

    if filename:
        plt.savefig(f"content/posts/sutton-barto/ucb/{filename}.svg", bbox_inches="tight")
        plt.savefig(f"content/posts/sutton-barto/ucb/{filename}.pdf", bbox_inches="tight")

    plt.show()

plot_and_save(ucb_optimal_action, eps_optimal_action, r"UCB ($c=1$)", r"$\epsilon$-greedy ($\epsilon=0.1$)", 
              ylabel="% Optimal Action", filename="ucb_optimal_action")

plot_and_save(ucb_reward, eps_reward, r"UCB ($c=1$)", r"$\epsilon$-greedy ($\epsilon=0.1$)", 
              ylabel="Average reward", filename="ucb_reward")



# plt.figure(figsize=(12,4.5))

# plt.xlabel("Steps", fontsize=16)
# plt.ylabel("% Optimal Action", fontsize=16)

# plt.plot(ucb_optimal_action[:100] * 100, label=r"$\UCB ($\c=2$)")
# plt.plot(eps_optimal_action[:100] * 100, label=r"$\epsilon$-greedy ($\epsilon=0.1$)")
# plt.legend(fontsize=14)
# plt.savefig("content/posts/sutton-barto/ucb/ucb_zoomed.svg", bbox_inches="tight")
# plt.savefig("content/posts/sutton-barto/ucb/ucb_zoomed.pdf", bbox_inches="tight")
# plt.show()
