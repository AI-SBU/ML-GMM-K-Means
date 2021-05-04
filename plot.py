import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")

'''
The function below takes in 3 dataset as parameters and constructs
scatter plots. All of the scatter plots are within the same figure.
Subplot was used to achieve this.
'''


def scatter_sub_plot(_data_one, _data_two, _data_three):
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


def hist_sub_plot(_data):
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


class Plot:
    def __init__(self, df, title, color):
        self._df = df
        self._title = title
        self._color = color

    def scatter_plot(self):
        plt.figure(figsize=(10, 8))
        plt.axvline(0, color="gray", ls="--")
        plt.axhline(0, color="gray", ls="--")
        plt.scatter(x=self._df.X1, y=self._df.X2, c=self._color, cmap="Spectral", edgecolors="black")
        plt.colorbar()
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(self._title)
        plt.show()
