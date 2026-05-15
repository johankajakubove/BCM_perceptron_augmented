import numpy as np
import json
import plot_changes
from plot_changes import plot_constants

THRESHOLD_STABILIZATION_MIN=10


class ClassicPerceptron:
    def __init__(self, layers, activation_functions, learning_rate, threshold_speed=0.0075, num=0.5, exp=0.4, init_w_mean=0.0, init_w_variance=0.02, plot=False):

        self.arch = layers
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.init_weight_mean = init_w_mean
        self.init_weight_variance = init_w_variance
        # make weights: notice +1 for bias, we have to keep in mind to add the bias to all computations
        self.weights_input_hidden = np.random.normal(self.init_weight_mean, self.init_weight_variance,
                                                     (self.arch[1], self.arch[0] + 1))
        self.weights_hidden_output = np.random.normal(self.init_weight_mean, self.init_weight_variance,
                                                      (self.arch[2], self.arch[1] + 1))

        self.threshold_speed = threshold_speed
        self.num = num
        self.exp = exp
        # watch weights
        self.weights_in_time_input_hidden = None
        self.weights_in_time_hidden_output = None
        # watch activations
        self.q_hidden = None
        self.q_hidden_means = None
        self.q_output = None
        self.q_output_means = None
        # watch learning rate
        self.watch_LR = None

        self.deltas_hidden = None
        self.deltas_hidden_means = None
        self.deltas_output = None
        self.deltas_output_means = None

        self.epoch = None
        self.plot=plot


    def activation(self, act_input, i_train):
        biased_input = np.vstack([act_input, np.ones(len(act_input[0]))])
        net_input_hidden = np.dot(self.weights_input_hidden, biased_input)
        act_hidden = self.activation_functions[0].apply_func(
            net_input_hidden
        )
        biased_act_hidden = np.vstack([act_hidden, np.ones(len(act_hidden[0]))])
        net_hidden_output = np.dot(self.weights_hidden_output, biased_act_hidden)
        act_output = self.activation_functions[1].apply_func(
            net_hidden_output
        )

        # --- watch NETS (x*y) ----
        if i_train:
            self.q_hidden[i_train] = biased_act_hidden.transpose() ** 2
            self.q_output[i_train] = act_output.transpose() ** 2
        return act_hidden, act_output

    def learn(self, act_input, act_hidden, act_output, labels, i_weights):
        batchsize = len(act_input[0])
        biased_act_input = np.vstack([act_input, np.ones(batchsize)])
        biased_act_hidden = np.vstack([act_hidden, np.ones(batchsize)])

        # activation(labels - output)
        delta_output = (labels - act_output) * self.activation_functions[1].apply_derived(act_output)
        # activation(labels - output) * weight * current activation
        delta_hidden = np.dot(self.weights_hidden_output[:, :-1].transpose(), delta_output) * \
                       self.activation_functions[0].apply_derived(act_hidden)
        self.deltas_hidden[i_weights[1]] = delta_hidden.transpose()
        self.deltas_output[i_weights[1]] = delta_output.transpose()

        # delta *= y
        weight_change_output = np.dot(delta_output, biased_act_hidden.transpose())
        # delta *= x
        weight_change_hidden = np.dot(delta_hidden, biased_act_input.transpose())

        # delta *= alpha (and include batch size)
        self.weights_input_hidden += (self.learning_rate / batchsize) * weight_change_hidden
        self.weights_hidden_output += (self.learning_rate / batchsize) * weight_change_output

        # --- watch WEIGHTS ----
        self.weights_in_time_input_hidden[i_weights[0]][i_weights[1]] = self.weights_input_hidden
        self.weights_in_time_hidden_output[i_weights[0]][i_weights[1]] = self.weights_hidden_output

        return True

    def training_cycle(self, max_epoch, data_train, labels_train, data_test, labels_test, minibatch_size):
        performance = dict({"train": [], "test": []})
        train_data_size = len(data_train[0])
        last_epoch = None

        self.q_hidden = np.zeros(shape=(train_data_size,self.arch[1] + 1))
        self.q_hidden_means = np.zeros(shape=(max_epoch, self.arch[1] + 1))
        self.q_output = np.zeros(shape=(train_data_size, self.arch[2]))
        self.q_output_means = np.zeros(shape=(max_epoch, self.arch[2]))

        self.weights_in_time_input_hidden = np.zeros(shape=(max_epoch, train_data_size, self.arch[1], self.arch[0] + 1))
        self.weights_in_time_hidden_output = np.zeros(shape=(max_epoch, train_data_size, self.arch[2], self.arch[1] + 1))

        self.deltas_hidden = np.zeros(shape=(train_data_size, self.arch[1]))
        self.deltas_hidden_means = np.zeros(shape=(max_epoch, self.arch[1]))
        self.deltas_output = np.zeros(shape=(train_data_size, self.arch[2]))
        self.deltas_output_means = np.zeros(shape=(max_epoch, self.arch[2]))

        self.watch_LR = np.zeros(shape=max_epoch)

        for epoch in range(max_epoch):
            self.epoch = epoch
            index_permutation = np.random.permutation(train_data_size)
            train_score = 0

            for i in range(0, train_data_size, minibatch_size):
                # take train data for current batch
                input_batch = data_train[:, index_permutation[i:i + minibatch_size]]
                label_batch = labels_train[:, index_permutation[i:i + minibatch_size]]

                # compute output and adjust weights
                act_hidden, act_output = self.activation(input_batch, i_train=i)
                self.learn(input_batch, act_hidden, act_output, label_batch, i_weights=(epoch, i))

                # get index of activated (the highest value) neuron
                train_output_winners = np.argmax(act_output, axis=0)
                train_target_winners = np.argmax(label_batch, axis=0)

                # get result
                train_score += np.sum(train_output_winners == train_target_winners)

            train_score = train_score / train_data_size
            performance["train"].append(train_score)
            test_score = self.test(data_test, labels_test)
            performance["test"].append(test_score)
            # print("Epoch: {}. Accuracy Train: {:.3f}% Test: {:.3f}%".format(epoch + 1, train_score * 100.,
            #                                                                 test_score * 100.))

            # calculate averages for watching weights/activations/deltas in current epoch
            self.q_hidden_means[epoch] = np.mean(self.q_hidden, axis=0)
            self.q_output_means[epoch] = np.mean(self.q_output, axis=0)
            self.deltas_hidden_means[epoch] = np.mean(self.deltas_hidden, axis=0)*100 # DELTA TIMES 100
            self.deltas_output_means[epoch] = np.mean(self.deltas_output, axis=0)*100

            # GLOBAL LR
            self.learning_rate = self.threshold_alteration(self.learning_rate, self.num, self.exp)
            self.watch_LR[self.epoch] = self.learning_rate

            # --- STOP TRAINING if test accuracy 100% ---
            # if abs(test_score * 100. - 100) < 0.0001:
            #     last_epoch = epoch + 1
            #     self.save_to_file()
            #     break

        if not last_epoch:
            last_epoch = max_epoch

        # calculate averages for watching weights/nets in current epoch
        self.weights_in_time_input_hidden =  np.mean(self.weights_in_time_input_hidden, axis=1)
        self.weights_in_time_hidden_output =  np.mean(self.weights_in_time_hidden_output, axis=1)

        if self.plot:
            plot_constants(self.watch_LR, last_epoch, self.num, self.exp)

        return performance, last_epoch

    def test(self, data_test, labels_test, print_result=False):
        _, output_test = self.activation(data_test, i_train=None)
        test_output_winners = np.argmax(output_test, axis=0)
        test_target_winners = np.argmax(labels_test, axis=0)
        test_score = np.sum(test_output_winners == test_target_winners) / len(data_test[0])
        if print_result:
            print("Accuracy  Test: {:.3f}%".format(test_score * 100.))
        return test_score

    def threshold_alteration_linear(self, x):
        if x-self.threshold_speed < 0:
            return x
        return x-self.threshold_speed

    def threshold_alteration(self, x, numerator=0.5, exponent=0.4):
        return numerator / ((self.epoch + 1) ** exponent)

    def plot_one(self, j, i, last_epoch):
        plot_changes.plot(j, i, self.weights_in_time_hidden_output, self.q_output_means,
                          self.watch_LR, False, self.learning_rate, self.arch[1],
                          last_epoch, homeostatic=False)

    def plot_all(self, last_epoch):
        plot_changes.plot_all(self.weights_in_time_input_hidden, self.q_hidden_means,
                                  self.watch_LR, self.deltas_hidden_means, True, self.arch, last_epoch, homeostatic=False)
        plot_changes.plot_all(self.weights_in_time_hidden_output, self.q_output_means,
                                  self.watch_LR, self.deltas_output_means, False, self.arch, last_epoch, homeostatic=False)


    def stabilize_thresholds(self):
        # ---- STOP AT STABILIZED NET ----
        n = THRESHOLD_STABILIZATION_MIN

        last_10_input_hidden = self.q_hidden_means[self.epoch-n:self.epoch]
        rounded_input_hidden = np.round(last_10_input_hidden, 3)
        all_same_mask = np.all(rounded_input_hidden == rounded_input_hidden[0], axis=0)

        for i in range(self.arch[1]):
            for j in range(self.arch[0] + 1):
                if bool(all_same_mask[i, j]) and self.learning_rates_hidden[i][j] != 0:
                    print(f"Threshold input_hidden [{i}][{j}] set to 0")
                    self.learning_rates_hidden[i][j] = 0

        last_10_hidden_output = self.q_output_means[self.epoch-n:self.epoch]
        rounded_hidden_output = np.round(last_10_hidden_output, 3)
        all_same_mask = np.all(rounded_hidden_output == rounded_hidden_output[0], axis=0)

        for i in range(self.arch[2]):
            for j in range(self.arch[1] + 1):
                if bool(all_same_mask[i, j]) and self.learning_rates_output[i][j] != 0:
                    print(f"Threshold hidden_output [{j}][{i}] set to 0")
                    self.learning_rates_output[i][j] = 0


    def make_one_hot(input_array):
        output_array = np.zeros((input_array.size, input_array.max()+1))
        output_array[np.arange(input_array.size),input_array] = 1
        return output_array

    def save_to_file(self, file_path='trained_hp'):
        np.savez(file_path, weights_hidden=self.weights_input_hidden, weights_output=self.weights_hidden_output,
                            thresholds_hidden=self.learning_rates_hidden, thresholds_output=self.learning_rates_output)

    def load_from_file(self, file_path='trained_hp'):
        data = np.load(file_path+'.npz')
        self.weights_input_hidden = data["weights_hidden"]
        self.weights_hidden_output = data["weights_output"]
        self.learning_rates_hidden = data["thresholds_hidden"]
        self.learning_rates_output = data["thresholds_output"]

    def make_one_hot(input_array):
        output_array = np.zeros((input_array.size, input_array.max()+1))
        output_array[np.arange(input_array.size),input_array] = 1
        return output_array