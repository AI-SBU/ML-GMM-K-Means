import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from plot import Plot


class TwoFeatureClustering:
    def __init__(self, _df, _cluster_range):
        self.df = _df
        self.cluster_range = _cluster_range

    # standardizing the data
    def standardize(self):
        z = StandardScaler()
        self.df = pd.DataFrame(z.fit_transform(self.df))
        self.df.rename(columns={0: "X1", 1: "X2"}, inplace=True)

    '''
    The function below uses the Elbow Method algorithm to determines 
    the number of optimal clusters given a dataset
    '''

    def elbow_method_kmeans(self):
        # sum of squares within clusters
        sswc = []
        # creates a model with clusters corresponding to each element in
        # cluster_range and appends their inertia to the list sswc
        for n_clusters in self.cluster_range:
            _ = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
            _.fit(self.df)
            sswc.append(_.inertia_)

        # plot
        plt.plot(self.cluster_range, sswc)
        plt.title("Elbow Method to Estimate Number of Clusters KMeans Would Use")
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSWC")
        plt.show()

    def bic_method_gmm(self):
        models = [GaussianMixture(clusters, random_state=42).fit(self.df) for clusters in self.cluster_range]
        plt.plot(self.cluster_range, [m.bic(self.df) for m in models], label="BIC")
        plt.legend(loc="best")
        plt.xlabel("Number of Clusters")
        plt.title("BIC Method to Estimate Number of Clusters GMM would Use")
        plt.show()

    def gmm_clustering(self):
        em = GaussianMixture(n_components=8, random_state=42)
        labels = em.fit_predict(self.df)
        Plot(self.df, "GMM Clusters : 8", color=labels).scatter_plot()

    '''
    The function below computes the silhouette coefficient values per clusters
    Plots each of the cluster against their silhouette coefficient values
    Plots the scatter plot of the each features
    Selecting the number of clusters with silhouette analysis on KMeans clustering — 
    scikit-learn 0.24.2 documentation, 2021)
    '''

    def kmeans_clustering(self):
        for n_clusters in self.cluster_range:
            # create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # ax1 is silhouette plot, setting x and y limits
            ax1.set_xlim([-1, 1])
            ax1.set_ylim([0, len(self.df) + (n_clusters + 1) * 10])

            # the clustering performed using GMM with n_clusters
            cluster = KMeans(n_clusters=n_clusters, random_state=42)
            labels = cluster.fit_predict(self.df)

            # calculate the silhouette score
            silhouette_avg = silhouette_score(self.df, labels)
            print("Number of clusters : ", n_clusters,
                  "-> Silhouette Score : ", silhouette_avg)

            # calculate silhouette score for each sample
            silhouette_per_sample = silhouette_samples(self.df, labels)
            y_lower = 10
            for i in range(n_clusters):
                # gather the silhouette score per samples belonging to the ith cluster
                ith_cluster_silhouette_values = silhouette_per_sample[labels == i]
                # sort them
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                cmap = cm.get_cmap("Spectral")
                color = cmap(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            cmap = cm.get_cmap("Spectral")
            colors = cmap(labels.astype(float) / n_clusters)
            ax2.scatter(self.df.X1, self.df.X2, marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = cluster.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')
                ax2.set_title("The visualization of the clustered data.")
                ax2.set_xlabel("Feature X1")
                ax2.set_ylabel("Feature X2")

                plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                              "with n_clusters = %d" % n_clusters),
                             fontsize=14, fontweight='bold')

        plt.show()


df = pd.read_csv("GMMData.csv")
raw_data = Plot(df, "Raw GMMData", df.X2)
# .scatter_plot()
cluster_range = np.arange(2, 10)
model = TwoFeatureClustering(df, cluster_range)
model.standardize()
# model.bic_method_gmm()
model.gmm_clustering()
# model.elbow_method_kmeans()
# model.kmeans_clustering()
