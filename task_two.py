from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")


def scatter_plot(_df, title):
    plt.figure(figsize=(10, 8))
    plt.axvline(0, color="gray", ls="--")
    plt.axhline(0, color="gray", ls="--")
    plt.scatter(x=df.X1, y=df.X2, c=df["p(X1)"], cmap="Spectral", edgecolors="black")
    plt.colorbar()
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.show()


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

scatter_plot(df, "GMM Cluster")
