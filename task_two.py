from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from plot import Plot
import pandas as pd
import seaborn as sns

sns.set_style("dark")


def contour_plot_3d():
    print("to do")


df = pd.read_csv("mvn_data_200_samples.csv")

# creating model with 2 clusters
em = GaussianMixture(n_components=2)
# fitting the model
em.fit(df)

cluster = em.predict(df)
cluster_prob = pd.DataFrame(em.predict_proba(df))

df["p(X1)"] = cluster_prob[0]
df["p(X2)"] = cluster_prob[1]

print(df)


Plot(df, "GMM Cluster", df["p(X1)"]).scatter_plot()
print("Silhouette Score", silhouette_score(df, cluster))
