from homeostatic_perceptron import *

class HomeostaticPerceptronActivation(HomeostaticPerceptron):
    def __init__(self, layers, activation_functions, learning_rate, theta, linear):
        super().__init__(layers, activation_functions, learning_rate, linear=linear)
        self.threshold = theta

    def _phi_with_thresholds(self, learning_rate, i, j, q, lr_change_start_array):
        lr_change_start = lr_change_start_array[i, j]
        if q > self.threshold or lr_change_start > 0:
            if lr_change_start != 0:
                lr_change_start_array[i, j] += 1
                return self.phi(lr_change_start + 1)
            lr_change_start_array[i, j] = 1
            return self.phi(1)
        return learning_rate

    def _phi_with_thresholds_linear(self, learning_rate, q):
        if q > self.threshold or learning_rate != self.learning_rate:
            return self.phi_linear(learning_rate)
        return learning_rate

    def BCM_thresholds(self, linear=False):
        i_idxs_hid, j_idxs_hid = np.indices(self.lr_change_start_hidden.shape)
        i_idxs_out, j_idxs_out = np.indices(self.lr_change_start_output.shape)

        if linear:
            vrctorized_phi = np.vectorize(self._phi_with_thresholds_linear)
            self.learning_rates_hidden = vrctorized_phi(self.learning_rates_hidden, self.q_hidden_means[self.epoch][i_idxs_hid])
            self.learning_rates_output = vrctorized_phi(self.learning_rates_output, self.q_output_means[self.epoch][i_idxs_out])

        else:
            def phi_hidden(lr, i, j, q):
                return self._phi_with_thresholds(lr, i, j, q, self.lr_change_start_hidden)

            vrctorized_phi = np.vectorize(phi_hidden)
            self.learning_rates_hidden = vrctorized_phi(self.learning_rates_hidden,
                                                        i_idxs_hid, j_idxs_hid, self.q_hidden_means[self.epoch][i_idxs_hid])

            def phi_output(lr, i, j, q):
                return self._phi_with_thresholds(lr, i, j, q, self.lr_change_start_output)

            vrctorized_phi = np.vectorize(phi_output)
            self.learning_rates_output = vrctorized_phi(self.learning_rates_output,
                                                        i_idxs_out, j_idxs_out, self.q_output_means[self.epoch][i_idxs_out])

    def plot(self, last_epoch):
        # deltas are set to None
        plot_changes.plot_all(self.weights_in_time_input_hidden, self.q_hidden_means,
                                  self.lr_watch_input_hidden, None, True, self.arch,
                                  last_epoch, homeostatic=True)
        plot_changes.plot_all(self.weights_in_time_hidden_output, self.q_output_means,
                                  self.lr_watch_hidden_output, None, False, self.arch,
                                  last_epoch, homeostatic=True)

