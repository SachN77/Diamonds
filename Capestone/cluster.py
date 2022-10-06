
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from random import randint


class Cluster:
    def __init__(self, X):
        self.X = X

    def fit(self, model):
        trained_model = model.fit(self.X)
        return
    
    def predict(self, model, type = ''):
        if type == 'H':
            y_pred = model.fit_predict(self.X)
        else:
            y_pred = model.predict(self.X)
        return y_pred

    # def score(self, model):
    #     model_score = model.score(self.X)
       
    #     return model_score

    def elbowCurve(self):
        from sklearn.cluster import KMeans
        scores = []
        #K = range(1, 11)
        #for k in K:
        for i in range(1, 11):
            model = KMeans(n_clusters=i, random_state = 0)
            #model.fit(x.reshape(-1,1))
            model.fit(self.X)
            scores.append(model.inertia_)
            
        plt.plot(range(1, 11), scores, 'bx-')
        #plt.plot(K, scores, 'bx-')
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.show()
        print('The optimal K is: 3')
        return scores

    def plotClusters(self, clusters, model):
        colors = ['orange', 'blue', 'green', 'magenta']
        plt.figure(figsize=(15,10))
        for i in range(3):
            plt.scatter(self.X[clusters == i, 0], self.X[clusters == i, 1], c=colors[i], label=f'Cluster {i + 1}')
            #plt.scatter(x[clusters == i], x[clusters == i], c=colors[i])
            plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], color='red', marker='+', s=100)

        plt.title('K-Means Clustering')
        plt.xlabel('')
        plt.ylabel('')
        plt.legend(loc='upper right')
        plt.show()
    
    def plot_hierarchicalClustering(self, y_pred):
        #plt.scatter(self.X, y_pred )
        plt.scatter(self.X[:, 0], self.X[:, 1], c=y_pred, cmap='rainbow')
        
        plt.show()
        
    def plot_dendrogram(self):
        import scipy.cluster.hierarchy as sch
        dendrogram = sch.dendrogram(sch.linkage(self.X, method='ward'), color_threshold = 120)
        plt.show()

    def inertia(self, model):
        inertia = model.inertia_
        return inertia

    def kmClustering(self):
        k = [2, 3, 5, 6, 7]
       
        from sklearn.cluster import KMeans
        self.elbowCurve()
        for i in k:
            kmeans_model = KMeans(n_clusters=i, random_state=0, tol = 0.01)
            self.fit(kmeans_model)
            kmeans_y_pred = self.predict(kmeans_model)
            inertia = self.inertia(kmeans_model)
            
        return

    def hierarchicalClustering(self):
        from sklearn.cluster import AgglomerativeClustering
        hc_model = AgglomerativeClustering(n_clusters = 5, linkage='ward')
        self.fit(hc_model)
        hc_y_pred = self.predict(hc_model, 'H')
        self.plot_hierarchicalClustering(hc_y_pred)
        #self.plot_dendrogram()

        return

    def meanshiftClustering(self):
        from sklearn.cluster import MeanShift
        ms_model = MeanShift(bandwidth = 1.0)
        self.fit(ms_model)
        ms_y_pred = self.predict(ms_model)
        self.plot_hierarchicalClustering(ms_y_pred)

        return

        #self.plotClusters(kmeans_y_pred, kmeans_model)
        #inertia = kmeans_model.inertia_