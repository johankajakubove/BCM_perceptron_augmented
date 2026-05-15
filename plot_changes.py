import numpy as np
import matplotlib.pyplot as plt

def plot(j, i, weights, nets, learning_rates, input, alpha, hidden_size, epochs, theta, homeostatic):
    curr_weight = weights[:epochs, j, i]
    curr_net = nets[:epochs, j]
    if homeostatic:
        curr_lr = learning_rates[:epochs, j, i]
    else:
        curr_lr = learning_rates[:epochs]

    title="Hidden-output"
    if input:
        title="Input-hidden"
    title += f" i: {i} j: {j}, learning rate: {alpha}, hidden size: {hidden_size}"

    # plt.title(title)
    plt.title(f"Linear decay schedule; theta: {theta}")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")

    plt.axhline(theta, color='lightgray', linewidth=1)

    # plt.ylim(-3, 6)
    x = range(1, epochs + 1)
    plt.plot(x, curr_weight, color="red", label="Weight")
    plt.plot(x, curr_net, color="blue", label="$\delta^2$*100")
    plt.plot(x, curr_lr, color="purple", label=r'$\alpha_n$')
    # plt.legend(["w", "delta", "theta"])
    plt.legend()

    # plt.savefig("example_2_5_actv_hyp.svg", format='svg', bbox_inches='tight', transparent=True)
    plt.show()


def plot_all(weights, q, learning_rates, deltas, input, size, epochs, homeostatic):
    # show weight progress in time INPUT-HIDDEN
    rows = size[1] if input else size[2]
    cols = size[0] + 1 if input else size[1] + 1

    x = range(1, epochs + 1)
    max_in_row = 2

    for j in range(rows):
        fig, axs = plt.subplots(nrows=cols//max_in_row + 1, ncols=max_in_row,
                                figsize=(max_in_row * 4, (cols//max_in_row + 1) * 5))
        for i in range(cols):
            ax = axs[i//max_in_row, i%max_in_row]
            curr_weight = weights[:epochs, j, i]

            if homeostatic:
                curr_lr = learning_rates[:epochs, j, i]
            else:
                curr_lr = learning_rates[:epochs]

            line1, = ax.plot(x, curr_weight, color="red", label="w")
            line2, = ax.plot(x, curr_lr, color="purple", label="alpha")
            # ax.set_ylim(-2, 12)
            ax.set_title(f"Neuron ({i},{j})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            handles = [line1, line2]

            if deltas is not None:
                curr_delta = deltas[:epochs, j]
                line4, = ax.plot(x, curr_delta, color="green", label="delta*100")
                handles.append(line4)

            if q is not None:
                curr_act = q[:epochs, j]
                line3, = ax.plot(x, curr_act, color="blue", label="$q^2$")
                handles.append(line3)

            ax.legend(handles=handles,fontsize="x-small")

        plt.suptitle("Input-hidden" if input else "Hidden-output")
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def plot_constants(learning_rates, epochs, num, exp):
    curr_lr = learning_rates[:epochs]
    plt.title("LR decay schedule;"+rf'$\alpha_n = \frac{{{num}}}{{(1+\mathrm{{epoch_n}})^{{{exp}}}}}$')
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")

    # plt.ylim(-3, 6)
    x = range(1, epochs + 1)
    plt.plot(x, curr_lr, color="purple", label=r'$\alpha_n$')
    # plt.legend(["w", "delta", "theta"])
    plt.legend()

    plt.savefig(f"1b_{num}_{exp}.svg", format='svg', bbox_inches='tight', transparent=True)
    plt.show()