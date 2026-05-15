import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    # Sigmoid: y = f(net) = 1 / (1+e^-net)
    def apply_func(self, net):
        return 1.0 / (1.0 + np.exp(-net))

    # Sigmoid derivation
    def apply_derived(self, output):
        return output * (1 - output)


class SoftMax:
    def __init__(self):
        pass

    def apply_func(self, net):
        e_net = np.exp(net)
        e_denom0 = e_net.sum(axis=0, keepdims=True)
        result = e_net / e_denom0
        return result

    def apply_derived(self, output):
        return output * (1 - output)