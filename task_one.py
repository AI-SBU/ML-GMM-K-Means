from gen_data import MVN
from plot import scatter_sub_plot
from plot import hist_sub_plot
import numpy as np
import pandas as pd

import seaborn as sns

sns.set_style("dark")


# initializing the parameters
mu_one = np.array([-1.0, -1.5])
cov_one = np.array([[1.0, 0.2], [0.2, 1.0]])
mu_two = np.array([1.0, 1.5])
cov_two = np.array([[2.0, 0.1], [0.1, 2.0]])
n_samples_one = 100
n_samples_two = 20


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
combined_class.to_csv("mvn_data_120_samples.csv", index=False, header=True)

scatter_sub_plot(class_one, class_two, combined_class)
hist_sub_plot(combined_class)
