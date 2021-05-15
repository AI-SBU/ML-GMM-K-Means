from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from plot import Plot
import pandas as pd

# Reading in the dataset used for task 3 and 5
df1 = pd.read_csv("mvn_data_200_samples.csv")
df2 = pd.read_csv("mvn_data_120_samples.csv")

'''
The function below takes in a dataset(dataframe) as a parameter
Performs clustering using the K-means algorithm
'''


def task3_5(df):
    # defining the number of clusters we want, in this scenario
    # we want two clusters for the two classes
    km = KMeans(n_clusters=2)
    # fitting and predicting
    km.fit(df)
    predict = km.predict(df)

    df["Predict"] = predict

    # generating a scatter plot from the predicted values
    Plot(df, "KMeans Clustering", predict).scatter_plot()
    # evaluation metric for the model
    print("Silhouette Score", silhouette_score(df, predict))


task3_5(df2)
