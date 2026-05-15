import math

from PIL import Image
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from homeostatic_perceptron_q import HomeostaticPerceptronActivation
from homeostatic_perceptron_delta import HomeostaticPerceptronDelta
from classic_perceptron import ClassicPerceptron
from plot_parameters import plot, plot_3d
from activation_functions import Sigmoid, SoftMax
import matplotlib.pyplot as plt

features = 4
classes = 3
max_epoch = 50
hidden_neurons = 9
learning_rate = 0.4
iterations = 20

iris = datasets.load_iris()
iris_inputs = pd.DataFrame(iris.data)
iris_labels = pd.DataFrame(iris.target)

data_train_pd, data_test_pd, target_train_pd, target_test_pd = train_test_split(iris_inputs, iris_labels,
                                                                                test_size=0.1,
                                                                                shuffle=True,
                                                                                random_state=123)
data_train = data_train_pd.values.transpose()
target_train = ClassicPerceptron.make_one_hot(target_train_pd.values.transpose()).transpose()
data_test = data_test_pd.values.transpose()
target_test = ClassicPerceptron.make_one_hot(target_test_pd.values.transpose()).transpose()


learning_rates = [0, 0.05, 0.1, 0.4, 0.5, 0.7]
hidden_sizes = [3, 6, 9, 12, 16, 24, 32]

sigmoid = Sigmoid()
softmax = SoftMax()

def run(perceptron, theta=None, save=False):
    if theta:
        network = perceptron([features, hidden_neurons, classes], [sigmoid, softmax], learning_rate, theta, False)
    else:
        network = perceptron([features, hidden_neurons, classes], [sigmoid, softmax], learning_rate)
    performance, last_epoch = network.training_cycle(max_epoch, data_train, target_train, data_test, target_test, 1)
    if save:
        network.save_to_file()
    return performance, last_epoch

# --- just test saved network ---
def run_test_from_file():
    network = HomeostaticPerceptronDelta([features, hidden_neurons, classes], [sigmoid, softmax], learning_rate)
    network.load_from_file()
    network.test(data_test, target_test, print_result=True)

def run_test_learning_rates():
    results = {}
    for lr in learning_rates:
        lr = round(lr, 2)
        performance_sum = 0
        results[lr] = np.zeros(iterations)
        for i in range(iterations):
            network = ClassicPerceptron([features, hidden_neurons, classes], [sigmoid, softmax], lr)
            performance, last_epoch = network.training_cycle(max_epoch, data_train, target_train, data_test, target_test, 1)
            results[lr][i] = performance['test'][max_epoch - 1]
    plot(results, iterations)

def test_hidden_sizes_and_lr(learning_rates):
    results = {}
    for lr in learning_rates:
        result = np.zeros(shape=(len(hidden_sizes), 3))
        for row, hids in enumerate(hidden_sizes):
            performance_sum = np.zeros(iterations)
            for i in range(1):
                print("Training learning rate {}. hidden size {}. Iter {}".format(lr,hids,i))
                network = ClassicPerceptron([features, hids, classes], [sigmoid, softmax], lr)
                performance, last_epoch = network.training_cycle(max_epoch, data_train, target_train, data_test, target_test, 1)
                performance_sum[i] = performance['test'][max_epoch-1]

            result[row] = [int(hids), round(np.mean(performance_sum), 3), round(np.std(performance_sum), 3)]
        np.savetxt(f"csv_data/data{lr}.csv", result, delimiter=",", header="x,y,err", comments='')
    print(results)
    plot_3d(hidden_sizes, learning_rates, results)

def test_Clasical():
    performance_sum = 0
    reached_100 = 0
    reached_93 = 0
    for i in range(iterations):
        network = ClassicPerceptron([features, hidden_neurons, classes], [sigmoid, softmax], learning_rate)
        performance, last_ep = network.training_cycle(max_epoch, data_train, target_train, data_test, target_test, 1)
        # print("Iter {} performance {}".format(i, performance['test'][last_epoch-1]))
        perf = round(performance['test'][last_ep - 1], 4) * 100
        if perf == 100:
            reached_100 += 1
        if perf == 93.33:
            reached_93 += 1
        performance_sum += performance['test'][last_ep - 1]
    results = performance_sum / iterations
    print("CLASSIC")
    print(results)
    print(reached_100)

def test_Clasical_constants(constants):
    for num,exp in constants:
        performance_sum = 0
        reached_100 = 0
        reached_93 = 0
        for i in range(iterations):
            network = ClassicPerceptron([features, hidden_neurons, classes], [sigmoid, softmax],
                                        learning_rate, num=num, exp=exp, plot=True if i == 0 else False)
            performance, last_ep = network.training_cycle(max_epoch, data_train, target_train, data_test, target_test, 1)
            # print("Iter {} performance {}".format(i, performance['test'][last_epoch-1]))
            perf = round(performance['test'][last_ep - 1], 4) * 100
            if perf == 100:
                reached_100 += 1
            if perf == 93.33:
                reached_93 += 1
            performance_sum += performance['test'][last_ep - 1]
        results = performance_sum / iterations
        print("CLASSIC", num, exp)
        print(results)
        print(reached_93)


def test_threhold(perceptron, thetas, linear=True, filename="output"):
    final = dict()
    for theta in thetas:
        performance_sum = 0
        performance_squared_sum = 0
        reached_100 = 0
        outreached_93 = 0
        reached_93 = 0
        for i in range(iterations):
            network = perceptron([features, hidden_neurons, classes], [sigmoid, softmax], learning_rate, theta, linear)
            performance, last_ep = network.training_cycle(max_epoch, data_train, target_train, data_test, target_test, 1)
            perf = round(performance['test'][last_ep-1], 4) * 100
            if perf > 93.33:
                outreached_93 += 1
            if perf == 100:
                reached_100 += 1
            if perf == 93.33:
                reached_93 += 1
            performance_sum += perf
            performance_squared_sum += perf ** 2

        mean_perf = performance_sum / iterations
        variance = (performance_squared_sum / iterations) - (mean_perf ** 2)
        std_dev = math.sqrt(variance) if variance > 0 else 0
        results = (round(mean_perf, 4), round(std_dev, 4))
        print("THETA:", theta)
        print("     ",results)
        print("     reached 100: ",reached_100)
        print("     outreached 93: ",outreached_93)
        print("     reached 93: ",reached_93)
        final[theta] = results
    print('________RESULTS________')
    print(final)

    with open(filename+".txt", "w") as f:
        f.write(f"{'theta':<6} {'mean':<5} {'stddev'}\n")
        for theta in sorted(final):
            mean, std = final[theta]
            mean_str = f"{mean:.2f}"
            std_str = f"{std:.2f}"
            f.write(f"{theta:<6} {mean_str:<7} {std_str}\n")


def plot_accuracy(performance, last_epoch):
    plt.plot(list(range(last_epoch)), performance["train"], label="Train")
    plt.plot(list(range(last_epoch)),performance["test"], label="Test")
    plt.title("Accuracy")
    plt.legend()
    # plt.savefig("loc_lr_const.svg", format='svg', bbox_inches='tight', transparent=True)
    plt.show()


def random_alphas_load():
    performance_sum = 0
    for i in range(iterations):
        network = HomeostaticPerceptronDelta([features, hidden_neurons, classes], [sigmoid, softmax], learning_rate, 0.2, True)
        network.load_from_file()
        print(data_test)
        performance_sum += network.test(data_test, target_test, print_result=True)
    results = performance_sum / iterations
    print(results)

performance, last_epoch = run(HomeostaticPerceptronDelta, 0.6)
plot_accuracy(performance, last_epoch)

init = [0, 1, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40]
leadres_lin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
leaders_hyp = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

deltas=[0.4,0.45,0.5,0.55,0.6]
# test_threhold(HomeostaticPerceptronDelta, deltas, False)
# test_Clasical_constants([(0.4,0.4)])