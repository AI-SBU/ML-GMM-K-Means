from gen_data import MVN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

sns.set_style("dark")

'''
The function below takes in 3 dataset as parameters and constructs
scatter plots. All of the scatter plots are within the same figure.
Subplot was used to achieve this.
'''


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


'''
The function below takes in a dataset as a parameter and constructs
histograms of the dataset as well as each of the columns in that dataset.
All of the histograms are within the same figure. Subplot was used to achieve this.
'''


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


# initializing the parameters
mu_one = np.array([-1.0, -1.5])
cov_one = np.array([[1.0, 0.2], [0.2, 1.0]])
mu_two = np.array([1.0, 1.5])
cov_two = np.array([[2.0, 0.1], [0.1, 2.0]])
n_samples_one = 100
n_samples_two = 100

# generating the datasets from MVN
class_one = pd.DataFrame(MVN(mu_one, cov_one, n_samples_one).gen_mvn_data())
class_two = pd.DataFrame(MVN(mu_two, cov_two, n_samples_two).gen_mvn_data())

# concatenating two datasets
combined_class = class_one.append(class_two, sort=False, ignore_index=True)

# assigning column names
combined_class.columns = ["X1", "X2"]
class_one.columns = ["X1", "X2"]
class_two.columns = ["X1", "X2"]

# writing the combined data of both classes to a file
combined_class.to_csv("mvn_data_200_samples.csv", index=False, header=True)

scatter_plot(class_one, class_two, combined_class)
hist_plot(combined_class)
