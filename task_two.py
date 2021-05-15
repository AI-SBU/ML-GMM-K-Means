from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from plot import Plot
import pandas as pd
import seaborn as sns

sns.set_style("dark")

# Reading the datasets used for task 2 and 4
df1 = pd.read_csv("mvn_data_200_samples.csv")
df2 = pd.read_csv("mvn_data_120_samples.csv")

'''
The function below takes in a dataset(dataframe) as a parameter
Performs clustering using the GMM
'''


def task2_4(df):
    # creating model with 2 clusters
    em = GaussianMixture(n_components=2)
    # fitting the model
    em.fit(df)

    # predicting the data
    cluster = em.predict(df)
    # getting the probabilities
    cluster_prob = pd.DataFrame(em.predict_proba(df))

    df["p(X1)"] = cluster_prob[0]
    df["p(X2)"] = cluster_prob[1]

    # Generating a scatter plot
    Plot(df, "GMM Cluster", df["p(X1)"]).scatter_plot()
    # Evaluation metric for the model
    print("Silhouette Score", silhouette_score(df, cluster))


task2_4(df2)
# Plot(df1, "Raw Data with 200 samples", color="gray").scatter_plot()
# Plot(df2, "Raw Data with 120 samples", color="gray").scatter_plot()
