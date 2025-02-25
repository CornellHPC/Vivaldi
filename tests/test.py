import sys
import time

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.cluster import KMeans

def read_data(fname):
    path = "../data/" + fname + ".txt"
    data = []
    labels = []

    # this currently assumes no missing data, 
    # all formatted with exactly one space between features
    with open(path, 'r') as file:
        rows = [line.rstrip() for line in file]
        for features in rows:
            features = features.split(' ')

            # extract label, not really needed tbh
            label = features.pop(0)
            labels.append(label)

            features = np.float32(
                list(map(lambda feature: float(feature.split(':')[1]), features)))
            data.append(features)
    
    data = np.array(data)

    # dump data as binary
    # out = open("../data/" + fname, 'wb')
    # data.tofile(out)
    # out.close()

    return data

def polynomial_kernel(data, gamma, c, r):
    B = data @ data.T
    K = np.power(gamma * B + c, r)
    return K

def init_clusters(n, k):
    cluster_id = 0
    clusters = np.zeros(n, dtype=int)  # which cluster does each point belong to
    clusters_size = np.zeros(k, dtype=int)  # cardinality of cluster i
    for i in range(n):
        clusters[i] = cluster_id
        clusters_size[cluster_id] += 1
        cluster_id = (cluster_id + 1) % k
    return clusters, clusters_size

def construct_v(clusters, cluster_size, k, n):
    V = np.zeros((k, n))
    for i in range(n):
        cluster = clusters[i]
        V[cluster][i] = 1 / cluster_size[cluster]
    return V

def calculate_score(data, clusters, k):
    score = 0

    for i in range(k):
        cluster = data[clusters == i]
        centroid = np.mean(cluster, axis=0)
        dists = cluster - centroid
        norms = np.linalg.norm(dists, axis=1)
        score += np.sum(norms ** 2)

    return score

if __name__ == "__main__":
    fname = sys.argv[1]
    k = int(sys.argv[2])

    # Read Data
    data = read_data(fname)  # read data and construct matrix
    n = data.shape[0]

    # model = KMeans(n_clusters=k, random_state=0, n_init='auto')
    # model.fit(data)
    # print('sklearn kmeans inertia (not kernelized):', model.inertia_)

    start_time = time.time()

    # Construct Kernel Matrix and P tilde
    K = polynomial_kernel(data, 1, 1, 1)  # kernel matrix
    # print(K)
    P = np.tile(np.diag(K), (k, 1)).T  # P tilde

    # Initialize cluster assignments and V sparse matrix
    clusters, clusters_size = init_clusters(n, k)
    V = construct_v(clusters, clusters_size, k, n)
    V = csc_matrix(V)

    # Training Loop
    maxiter = 100
    last_C = None
    for iter in range(maxiter):
        E = -2 * (K @ V.T)
        z = -0.5 * np.array([E[i][clusters[i]] for i in range(n)])
        C = V @ z.T
        D = E + P + C

        new_clusters = np.argmin(D, axis=1)

        # Convergence check
        # if np.all(clusters == new_clusters):
        #     print('Converged after', iter + 1, 'iterations')
        #     break

        clusters = new_clusters
        clusters_size = np.zeros(k)
        for i in range(n):
            clusters_size[clusters[i]] += 1

        V = construct_v(clusters, clusters_size, k, n)
        V = csc_matrix(V)
        # print('Finished Iteration', iter)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    
    # score = calculate_score(data, clusters, k)
    # print('kernel kmeans score (sum of squared dists):', score)

    clusters = clusters.astype(np.int64)
    print(clusters)
    clusters.tofile("../data/" + fname + "_py")
