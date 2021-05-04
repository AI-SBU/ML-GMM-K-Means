from scipy import stats
from sklearn.mixture import GaussianMixture
from DataGen import MVN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

sns.set_style("dark")


def scatter_plot(_data_one, _data_two, _data_three):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=_data_one)
    plt.title("Class One")
    plt.xlabel("Range")

    plt.subplot(2, 2, 2)
    sns.scatterplot(data=_data_two)
    plt.title("Class Two")
    plt.xlabel("Range")

    plt.subplot(2, 2, 3)
    sns.scatterplot(data=_data_three)
    plt.title("Combined Classes")
    plt.xlabel("Range")

    plt.show()


def hist_plot(_data):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data=_data.X1, bins=25, kde=True)
    plt.title("Class One")

    plt.subplot(2, 2, 2)
    sns.histplot(data=_data.X2, bins=25, kde=True)
    plt.title("Class Two")

    plt.subplot(2, 2, 3)
    sns.histplot(data=_data, bins=25, kde=True)
    plt.title("Class One & Two")
    plt.xlabel("X1 & X2")

    plt.show()


def task_one():
    mu_one = np.array([-1.0, -1.5])
    cov_one = np.array([[1.0, 0.2], [0.2, 1.0]])
    mu_two = np.array([1.0, 1.5])
    cov_two = np.array([[2.0, 0.1], [0.1, 2.0]])
    n_samples_one = 100
    n_samples_two = 100
    class_one = pd.DataFrame(MVN(mu_one, cov_one, n_samples_one, 0).gen_mvn_data())
    class_two = pd.DataFrame(MVN(mu_two, cov_two, n_samples_two, 42).gen_mvn_data())

    combined_class = class_one.append(class_two, sort=False, ignore_index=True)
    combined_class.columns = ["X1", "X2"]
    class_one.columns = ["X1", "X2"]
    class_two.columns = ["X1", "X2"]

    print(combined_class.head())

    scatter_plot(class_one, class_two, combined_class)
    hist_plot(combined_class)


task_one()
