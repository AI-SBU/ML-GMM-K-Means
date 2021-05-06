from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from plot import Plot
import pandas as pd

df1 = pd.read_csv("mvn_data_200_samples.csv")
df2 = pd.read_csv("mvn_data_120_samples.csv")


def task3_5(df):
    km = KMeans(n_clusters=2)
    km.fit(df)
    predict = km.predict(df)

    df["Predict"] = predict

    Plot(df, "KMeans Clustering", predict).scatter_plot()
    print("Silhouette Score", silhouette_score(df, predict))


task3_5(df2)
