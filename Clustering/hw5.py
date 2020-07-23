#%%
# always import
import sys
from time import time

# numpy & scipy
import numpy as np
import scipy

# sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from sklearn import neighbors

# Hungarian algorithm
from munkres import Munkres
from scipy.optimize import linear_sum_assignment

# visuals
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.manifold import Isomap, TSNE

# maybe
from numba import jit
import random
#%%
# load MNIST data and normalization
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='mnist/')
y = np.asarray(list(map(int, y)))
X = np.asarray(X.astype(float))
X = scale(X)
n_digits = len(np.unique(y))
#%%
#Kmeans
#2(a)-i
#%%
def euclidean_distance(a, b):
    return np.linalg.norm(a-b)
#%%
class kmeans():

    def __init__(self, K= n_digits, maximum_iteration = 300, tau = 0.0001):
        self.K = K
        self.maximum_iteration = maximum_iteration
        self.tau = tau
        self.clusters = [[] for i in range (self.K)]
        self.centroids = []
        
    def fit(self, data):
        self.data = data
        self.samples = data.shape[0]
        self.feature = data.shape[1]
        self.centroids= self.initialize_centroid(data)
        previous_objective = 0
        iteration = 0
        for i in range (self.maximum_iteration):
            self.clusters = self.cluster_create(self.centroids)
            previous_centroid = self.centroids
            self.centroids = self.calculate_new_centroids(self.clusters)
            current_objective = self.calculate_objective_function(self.centroids)
            if abs(previous_objective - current_objective ) < self.tau:
                break
            previous_objective =current_objective
            iteration = iteration+1
        print(iteration)
        return self.cluster_label(self.clusters)

    def initialize_centroid(self, data):
        pca = PCA()
        pca = pca.fit(data)
        initial_centroid = pca.components_[:10]
        initial_centroid_list = list()
        for i in range(initial_centroid.shape[0]):
            initial_centroid_list.append(initial_centroid[i, :])
        return initial_centroid_list

    def cluster_create(self, centroids ):
        clusters = [[] for i in range(self.K)]
        for index, sample in enumerate(self.data):
            sample2centroid_distance = [euclidean_distance(sample, c) for c in centroids]
            clusters[np.argmin(sample2centroid_distance)].append(index)
        return clusters
    
    def calculate_objective_function(self, centroids):
        sum_square_distance = list()
        for index, sample in enumerate(self.data):
            sample2centroid_distance = [euclidean_distance(sample, c) for c in centroids]
            sum_square_distance.append(np.min(sample2centroid_distance))
        return np.sum(np.square(sum_square_distance))
        
    def calculate_new_centroids(self, clusters):
        centroids = np.zeros((self.K, self.feature))
        for index, cluster in enumerate(clusters):
            meanOfCluster = np.mean(self.data[cluster], axis=0)
            centroids[index] = meanOfCluster
        return centroids
    
    def cluster_label(self, clusters):
        labels = np.empty (self.samples)
        for index, cluster in enumerate(clusters):
            for item in cluster:
                labels[item] = index
        return labels
    
    def fit1(self, data):
        j= -1
        objective = [[] for i in range(10)]
        prediction = [[] for i in range(10)]
        for l in range (10):
            j= j+1
            self.data = data
            self.samples = data.shape[0]
            self.feature = data.shape[1]
            seedValue = random.randrange(sys.maxsize)
            random.seed(seedValue)
            random_sample = np.random.choice(self.samples, self.K, replace=False)
            self.centroids = [self.data[index] for index in random_sample]
            previous_objective = 0
            for i in range (self.maximum_iteration):
                self.clusters = self.cluster_create(self.centroids)
                previous_centroid = self.centroids
                self.centroids = self.calculate_new_centroids(self.clusters)
                current_objective = self.calculate_objective_function(self.centroids)
                if abs(previous_objective - current_objective ) < self.tau:
                    break
                previous_objective =current_objective
            objective[j].append(current_objective)
            prediction[j].append(self.cluster_label(self.clusters))   
        return objective, prediction

    def best_centroid_4KNN(self, data):
        self.data = data
        self.samples = data.shape[0]
        self.feature = data.shape[1]
        random_sample = np.random.choice(self.samples, self.K, replace=False)
        self.centroids = [self.data[index] for index in random_sample]
        previous_objective = 0
        for i in range (self.maximum_iteration):
            self.clusters = self.cluster_create(self.centroids)
            previous_centroid = self.centroids
            self.centroids = self.calculate_new_centroids(self.clusters)
            current_objective = self.calculate_objective_function(self.centroids)
            if abs(previous_objective - current_objective ) < self.tau:
                break
            previous_objective =current_objective
        return self.centroids
#%%
k = kmeans(K= n_digits, maximum_iteration = 300, tau = 0.0001)
y_pred = k.fit(X)
# %%
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
print("%.6f" % homogeneity_score(y, y_pred))
print("%.6f" % completeness_score(y, y_pred))
print("%.6f" % v_measure_score(y, y_pred))
print("%.6f" % adjusted_mutual_info_score(y, y_pred))
print("%.6f" % adjusted_rand_score(y, y_pred))
#%%
#2a(ii)
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=30)
X_dash = pca.fit_transform(X)
k = kmeans(K= n_digits, maximum_iteration = 300, tau = 0.0001)
ob, pred = k.fit1(X_dash)
smallest_objective = np.argmin(ob)
y_pred1 = pred[smallest_objective]
y_pred1 = np.array(y_pred1[0])
#%%
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
print("%.6f" % homogeneity_score(y, y_pred1))
print("%.6f" % completeness_score(y, y_pred1))
print("%.6f" % v_measure_score(y, y_pred1))
print("%.6f" % adjusted_mutual_info_score(y, y_pred1))
print("%.6f" % adjusted_rand_score(y, y_pred1))
#%%
#Hungarian starts
#%%
def cost_matrix(y_pred, y, k):
    size_y = len(y)
    init_cost_mat = np.zeros((k, k))
    for i in range (k):
        for j in range (k):
            bool_array = np.logical_and(y_pred == i, y == j)
            init_cost_mat[i][j] = np.sum(bool_array)
    return (1-init_cost_mat)

#%%
# y_pred1
mun= Munkres()
index = mun.compute(cost_matrix(y_pred1, y, n_digits))
mp = {prev: cur for (prev, cur) in index}
munkres_label = np.array([mp[i] for i in y_pred1 ])
cnf_mat = confusion_matrix(y, munkres_label, labels=range(n_digits))
accuracy = np.trace(cnf_mat, dtype=float) / np.sum(cnf_mat)

#%%
# y_pred
mun= Munkres()
index = mun.compute(cost_matrix(y_pred, y, n_digits))
mp = {prev: cur for (prev, cur) in index}
munkres_label = np.array([mp[i] for i in y_pred ])
cnf_mat = confusion_matrix(y, munkres_label, labels=range(n_digits))
accuracy = np.trace(cnf_mat, dtype=float) / np.sum(cnf_mat)        
#%%
#Spectral_Clustering_Starts
#%%
from sklearn.neighbors import NearestNeighbors
def data_initialization(X):
    pca = PCA(n_components=30)
    X_dash = pca.fit_transform(X)
    return X_dash

X_dash = data_initialization(X)

def sparse_distance(X_dash):
    E = NearestNeighbors(n_neighbors=500, algorithm= 'kd_tree', metric= 'euclidean')
    E= E.fit(X_dash).kneighbors_graph(mode = 'distance')
    return E

H = sparse_distance(X_dash)

#%%
from scipy import sparse
def sigma(H):
    sum_H = H.sum()
    Nh= H.nonzero()
    H_mod = len(Nh[0])
    sig = sum_H/H_mod
    return sig

sig = sigma(H)
#%%
from scipy import sparse

def similarity_matrix(H, sig):
    np.square(H.data, out = H.data)
    sig2 = sig**2
    np.divide(H.data, sig2, out = H.data)
    np.exp(-(H.data), out= H.data)
    S = sparse.csr_matrix(1/H.sum(1))
    E = H.multiply(S)
    E.setdiag(1.0)
    ET = E.transpose()
    E= (E+ET)/2
    return E

E = similarity_matrix(H, sig)

# %%
from scipy import sparse

def degree_matrix(E):
    diag = np.squeeze(np.asarray(E.sum(1)))
    D= sparse.diags(diag, format='csr')
    return D

D = degree_matrix(E)

def calculate_laplacian(E, D):
    I = sparse.identity(E.shape[0])
    np.power(D.data, -0.5, out= D.data)
    DED = (D.dot(E)).dot(D)
    L= I-DED
    return L

L = calculate_laplacian(E, D)
#%%
from scipy.sparse.linalg import eigs
vals1, vecs1 = sparse.linalg.eigs(L, k= 20, which= 'SM')

def normalizing_eigenvector(vecs):
    vecs = vecs/np.linalg.norm(vecs, axis= 1).reshape(-1, 1)
    return vecs

vecs1 = normalizing_eigenvector(vecs1)
sorted_vecs = np.delete(vecs1, 0, 1).real
#%%
kmeans= KMeans(n_clusters=10, init='k-means++', n_init=10).fit(sorted_vecs)
mun= Munkres()
index = mun.compute(cost_matrix(kmeans.labels_, y, 10))
mp = {prev: cur for (prev, cur) in index}
munkres_label = np.array([mp[i] for i in kmeans.labels_])
cnf_mat = confusion_matrix(y, munkres_label, labels=range(10))
accuracy = np.trace(cnf_mat, dtype=float) / np.sum(cnf_mat)
#%%
#K_Nearest_Neighbour_Starts
#%%
class Knearest:
    def __init__(self, X, y, k):
        self.k = k
        self.X_train = X
        self.y_train = y
    def predict(self,X):
        y_pred = np.array([self.fit(sample) for sample in X])
        return y_pred
    def fit(self, X):
        nearest_neighbour_index = np.argsort([euclidean_distance(X, x_train) for x_train in self.X_train])[:self.k]
        nearest_neighbour_label = [self.y_train[sample] for sample in nearest_neighbour_index]  
        voting = Counter(nearest_neighbour_label).most_common(1)
        return voting [0][0]
def accuracy(true, pred):
    accuracy = np.sum(true == pred) / len(true)
    return accuracy
#%%
#Kmeans method
from sklearn.decomposition import PCA
pca = PCA(n_components=30)
X_dash = pca.fit_transform(X)
k = kmeans(K= 100, maximum_iteration = 300, tau = 0.0001)
centroids = k.best_centroid_4KNN(X_dash)
centroids = np.array(centroids)
X_dash = X_dash.tolist()
selected_X_dash = []
l1= list(range(70000))
l2 = []
for index, sample in enumerate(centroids):
    sample2centroid_distance = [euclidean_distance(c, sample) for c in X_dash]
    selected_X_dash.append(X_dash[np.argmin(sample2centroid_distance)])
    l2.append(np.argmin(sample2centroid_distance))
l3 = [x for x in l1 if x not in l2]
X_test = np.array([X_dash[l] for l in l3])
X_train = np.array(selected_X_dash)
y_test = np.array([y[l] for l in l3]) 
y_train = np.array([y[l] for l in l2]) 
from collections import Counter
Accuracy = []
for i in range(1, 6, 2):
    classification = Knearest(X = X_train, y= y_train, k= i)
    pred = classification.predict(X_test)
    acc = accuracy(y_test, pred)
    Accuracy.append(acc)

#%%
#Spectral clustering method 
from sklearn.decomposition import PCA
pca = PCA(n_components=30)
X_dash = pca.fit_transform(X)
H1 = sparse_distance(X_dash)
sig1= sigma(H1)
E1 = similarity_matrix(H1, sig1)
D1 = degree_matrix(E1)
L1 = calculate_laplacian(E1, D1)
vals1, vecs1 = sparse.linalg.eigs(L1, k= 20, which= 'SM')
vecs1 = normalizing_eigenvector(vecs1)
#%%
k1 = kmeans(K= 100, maximum_iteration = 300, tau = 0.0001)
centroids1 = k1.best_centroid_4KNN(sorted_vecs)
centroids1 = np.array(centroids1)
sorted_vecs = sorted_vecs.tolist()
selected_vecs1 = []
k1= list(range(70000))
k2 = []
for index, sample in enumerate(centroids1):
    sample2centroid_distance = [euclidean_distance(c, sample) for c in sorted_vecs]
    selected_vecs1.append(sorted_vecs[np.argmin(sample2centroid_distance)])
    k2.append(np.argmin(sample2centroid_distance))
k3 = [x for x in k1 if x not in k2]
X_test1 = np.array([sorted_vecs[k] for k in k3])
X_train1 = np.array(selected_vecs1)
y_test1 = np.array([y[k] for k in k3]) 
y_train1 = np.array([y[k] for k in k2]) 

#%%
from collections import Counter
Accuracy1 = []
for i in range(1, 6, 2):
    classification = Knearest(X = X_train1, y= y_train1, k= i)
    pred1 = classification.predict(X_test1)
    acc1 = accuracy(y_test1, pred1)
    Accuracy1.append(acc1)
#%%
#Random Sampling
from sklearn.decomposition import PCA
pca = PCA(n_components=30)
X_dash = pca.fit_transform(X)
sample2 = X_dash.shape[0]
random_sample2 = np.random.choice(sample2, 100, replace=False).tolist()
X_train2 = np.array([X_dash[index] for index in random_sample2])
m1= list(range(70000))
m2 = [x for x in m1 if x not in random_sample2]
X_test2 = np.array([X_dash[m] for m in m2])
y_test2 = np.array([y[k] for k in m2]) 
y_train2 = np.array([y[index] for index in random_sample2])
from collections import Counter
Accuracy2 = []
#%%
for i in range(1, 6, 2):
    classification = Knearest(X = X_train2, y= y_train2, k= i)
    pred2 = classification.predict(X_test2)
    acc2 = accuracy(y_test2, pred2)
    Accuracy2.append(acc2)



# %%
