from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from plot import Plot
import pandas as pd

df = pd.read_csv("mvn_data_200_samples.csv")

km = KMeans(n_clusters=2)
km.fit(df)
predict = km.predict(df)

df["Predict"] = predict

Plot(df, "KMeans Clustering", predict).scatter_plot()
print("Silhouette Score", silhouette_score(df, predict))
