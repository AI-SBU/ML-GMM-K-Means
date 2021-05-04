from sklearn.cluster import KMeans
from task_two import scatter_plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")

df = pd.read_csv("mvn_data_200_samples.csv")

km = KMeans(n_clusters=2)

km.fit(df)

predict = km.predict(df)

print(predict)

# scatter_plot(df, "KMeans Clustering")
