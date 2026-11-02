from datasets import load_iris_split
from experiments import run, plot_accuracy, experiment_learning_rates, experiment_classic, experiment_thresholds
from homeostatic_perceptron_delta import HomeostaticPerceptronDelta

HIDDEN_NEURONS = 9
LEARNING_RATE = 0.4
MAX_EPOCH = 50
ITERATIONS = 20

if __name__ == "__main__":
    iris = load_iris_split()

    performance, last_epoch = run(iris, HomeostaticPerceptronDelta, theta=0.6, linear=False,
                                  hidden_neurons=HIDDEN_NEURONS, learning_rate=LEARNING_RATE, max_epoch=MAX_EPOCH)
    plot_accuracy(performance, last_epoch)

    # old experimenta

    # architecture and learning rate, classical perceptron with constant learning rate
    # experiment_learning_rates(iris, learning_rates=[0, 0.05, 0.1, 0.4, 0.5, 0.7],
    #                           hidden_sizes=[3, 6, 9, 12, 16, 24, 32],
    #                           decay=False, max_epoch=MAX_EPOCH, iterations=ITERATIONS)

    # classical perceptron with decaying learning rate num / (epoch + 1)^exp
    # for num, exp in [(0.4, 0.4)]:
    #     experiment_classic(iris, decay=True, num=num, exp=exp, plot_first=True,
    #                        hidden_neurons=HIDDEN_NEURONS, learning_rate=LEARNING_RATE,
    #                        max_epoch=MAX_EPOCH, iterations=ITERATIONS)

    # homeostatic thresholds
    init = [0, 1, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40]
    leaders_lin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    leaders_hyp = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    deltas = [0.4, 0.45, 0.5, 0.55, 0.6]
    # experiment_thresholds(iris, HomeostaticPerceptronDelta, deltas, linear=False,
    #                       hidden_neurons=HIDDEN_NEURONS, learning_rate=LEARNING_RATE,
    #                       max_epoch=MAX_EPOCH, iterations=ITERATIONS)
