import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from classic_perceptron import ClassicPerceptron

IRIS_SPLIT_SEED = 123


def load_iris_split(split_seed=IRIS_SPLIT_SEED, test_size=0.1):
    iris = load_iris()
    iris_inputs = pd.DataFrame(iris.data)
    iris_labels = pd.DataFrame(iris.target)

    data_train_pd, data_test_pd, target_train_pd, target_test_pd = train_test_split(
        iris_inputs, iris_labels, test_size=test_size,shuffle=True,random_state=split_seed)
    data_train = data_train_pd.values.transpose()
    target_train = ClassicPerceptron.make_one_hot(target_train_pd.values.transpose()).transpose()
    data_test = data_test_pd.values.transpose()
    target_test = ClassicPerceptron.make_one_hot(target_test_pd.values.transpose()).transpose()
    return data_train, target_train, data_test, target_test
