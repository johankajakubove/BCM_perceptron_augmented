import math
import os
import numpy as np
import matplotlib.pyplot as plt
from classic_perceptron import ClassicPerceptron
from homeostatic_perceptron_delta import HomeostaticPerceptronDelta
from activation_functions import Sigmoid, SoftMax
from plot_parameters import plot_3d

SEED = 42   #for the starting weights and shuffling

SIGMOID = Sigmoid()
SOFTMAX = SoftMax()


def architecture(dataset, hidden_neurons):
    data_train, target_train, _, _ = dataset
    return [data_train.shape[0], hidden_neurons, target_train.shape[0]]


def one_mistake_accuracy(dataset):
    # accuracy in % with exactly one wrong test sample (93.33 for the 15 Iris test samples)
    _, _, data_test, _ = dataset
    n_test = data_test.shape[1]
    return round((n_test - 1) / n_test, 4) * 100


def run(dataset, perceptron, *, hidden_neurons, learning_rate, max_epoch, theta=None, linear=False,
        save=False, seed=SEED):
    data_train, target_train, data_test, target_test = dataset
    layers = architecture(dataset, hidden_neurons)
    if theta is not None:
        network = perceptron(layers, [SIGMOID, SOFTMAX], learning_rate, theta, linear, seed=seed)
    else:
        network = perceptron(layers, [SIGMOID, SOFTMAX], learning_rate, seed=seed)
    performance, last_epoch = network.training_cycle(max_epoch, data_train, target_train, data_test, target_test, 1)
    if save:
        network.save_to_file()
    return performance, last_epoch


def plot_accuracy(performance, last_epoch):
    plt.plot(list(range(last_epoch)), performance["train"], label="Train")
    plt.plot(list(range(last_epoch)),performance["test"], label="Test")
    plt.title("Accuracy")
    plt.legend()
    # plt.savefig("loc_lr_const.svg", format='svg', bbox_inches='tight', transparent=True)
    plt.show()


# just test saved network
def test_saved_network(dataset, *, hidden_neurons, perceptron=HomeostaticPerceptronDelta, file_path='trained_hp'):
    _, _, data_test, target_test = dataset
    # learning rate, theta and linear don't matter here: load_from_file replaces them with the saved values
    network = perceptron(architecture(dataset, hidden_neurons), [SIGMOID, SOFTMAX], 0, None, False)
    network.load_from_file(file_path)
    return network.test(data_test, target_test, print_result=True)


def experiment_learning_rates(dataset, learning_rates, hidden_sizes, *, decay, max_epoch, iterations):
    data_train, target_train, data_test, target_test = dataset
    results = {}
    os.makedirs("csv_data", exist_ok=True)
    for lr in learning_rates:
        result = np.zeros(shape=(len(hidden_sizes), 3))
        for row, hids in enumerate(hidden_sizes):
            accuracies = np.zeros(iterations)
            for i in range(iterations):
                print("Training learning rate {}. hidden size {}. Iter {}".format(lr, hids, i))
                network = ClassicPerceptron(architecture(dataset, hids), [SIGMOID, SOFTMAX], lr,
                                            decay=decay, seed=SEED + i)
                performance, last_epoch = network.training_cycle(max_epoch, data_train, target_train,
                                                                 data_test, target_test, 1)
                accuracies[i] = performance['test'][last_epoch - 1]

            results[(lr, hids)] = np.mean(accuracies)
            result[row] = [int(hids), round(np.mean(accuracies), 3), round(np.std(accuracies), 3)]
        np.savetxt(f"csv_data/data{lr}.csv", result, delimiter=",", header="x,y,err", comments='')
    plot_3d(hidden_sizes, learning_rates, results)
    return results


def experiment_classic(dataset, *, decay, hidden_neurons, learning_rate, max_epoch, iterations,
                       num=0.5, exp=0.4, plot_first=False):
    # baseline: classical perceptron; with decay=True the learning rate follows num / (epoch + 1)^exp
    data_train, target_train, data_test, target_test = dataset
    one_mistake = one_mistake_accuracy(dataset)
    performance_sum = 0
    reached_100 = 0
    reached_one_mistake = 0
    for i in range(iterations):
        network = ClassicPerceptron(architecture(dataset, hidden_neurons), [SIGMOID, SOFTMAX], learning_rate,
                                    num=num, exp=exp, plot=plot_first and i == 0, decay=decay, seed=SEED + i)
        performance, last_ep = network.training_cycle(max_epoch, data_train, target_train, data_test, target_test, 1)
        perf = round(performance['test'][last_ep - 1], 4) * 100
        if perf == 100:
            reached_100 += 1
        if perf == one_mistake:
            reached_one_mistake += 1
        performance_sum += performance['test'][last_ep - 1]
    results = performance_sum / iterations
    print("CLASSIC", f"decay num={num} exp={exp}" if decay else "constant learning rate")
    print("     mean accuracy: ", results)
    print("     reached 100: ", reached_100)
    print("     one mistake: ", reached_one_mistake)
    return results


def experiment_thresholds(dataset, perceptron, thetas, *, linear, hidden_neurons, learning_rate, max_epoch,
                          iterations, filename="output"):
    data_train, target_train, data_test, target_test = dataset
    one_mistake = one_mistake_accuracy(dataset)
    final = dict()
    for theta in thetas:
        performance_sum = 0
        performance_squared_sum = 0
        reached_100 = 0
        above_one_mistake = 0
        reached_one_mistake = 0
        for i in range(iterations):
            network = perceptron(architecture(dataset, hidden_neurons), [SIGMOID, SOFTMAX], learning_rate,
                                 theta, linear, seed=SEED + i)
            performance, last_ep = network.training_cycle(max_epoch, data_train, target_train, data_test, target_test, 1)
            perf = round(performance['test'][last_ep-1], 4) * 100
            if perf > one_mistake:
                above_one_mistake += 1
            if perf == 100:
                reached_100 += 1
            if perf == one_mistake:
                reached_one_mistake += 1
            performance_sum += perf
            performance_squared_sum += perf ** 2

        mean_perf = performance_sum / iterations
        variance = (performance_squared_sum / iterations) - (mean_perf ** 2)
        std_dev = math.sqrt(variance) if variance > 0 else 0
        results = (round(mean_perf, 4), round(std_dev, 4))
        print("THETA:", theta)
        print("     ",results)
        print("     reached 100: ",reached_100)
        print("     above one mistake: ",above_one_mistake)
        print("     one mistake: ",reached_one_mistake)
        final[theta] = results
    print('________RESULTS________')
    print(final)

    with open(filename+".txt", "w") as f:
        f.write(f"# seeds: {SEED}-{SEED + iterations - 1}\n")
        f.write(f"{'theta':<6} {'mean':<5} {'stddev'}\n")
        for theta in sorted(final):
            mean, std = final[theta]
            mean_str = f"{mean:.2f}"
            std_str = f"{std:.2f}"
            f.write(f"{theta:<6} {mean_str:<7} {std_str}\n")
    return final
