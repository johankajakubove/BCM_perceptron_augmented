import numpy as np
import matplotlib.pyplot as plt

def plot(params, epochs):
    lines = np.array([*params])
    x = range(1, 1+epochs)
    for p in lines:
        vals = params[p]
        plt.plot(x, vals, label=f"learing rate {p}")

    plt.title("Learning rates")
    plt.xlabel("Repetition")
    plt.ylabel("Performance")

    plt.ylim(0.5, 1)
    plt.legend()
    plt.show()


def plot_3d(hidden_neurons, learning_rates, accuracies):
    for lr in learning_rates:
        y = [accuracies.get((lr, hn), np.nan) for hn in hidden_neurons]
        plt.plot(hidden_neurons, y, marker='o', label=f"Lr={lr}")

    plt.xticks(hidden_neurons)
    plt.xlabel("Hidden Neurons")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Hidden Neurons")
    plt.legend()
    plt.grid(True)

    plt.savefig("experiment_1.svg", format='svg', bbox_inches='tight', transparent=True)

    plt.show()