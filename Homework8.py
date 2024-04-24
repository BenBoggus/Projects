import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dX = pd.read_csv('C:/Users/benbo/OneDrive/Documents/Code/Ck.csv', skipinitialspace=True)
Y = dX.values

def gonzalez_algorithm(X, K):
    centers = [X[0]]  # Choose the first point as the initial center
    while len(centers) < K:
        distances = np.array([min(np.linalg.norm(x - c) for c in centers) for x in X])
        new_center = X[np.argmax(distances)]
        centers.append(new_center)
    return np.array(centers)

def kmeans_plusplus(X, K):
    # Choose the first center randomly
    centers = [X[np.random.randint(len(X))]]
    
    # Loop until we have K centers
    while len(centers) < K:
        # Calculate distances of each point to the nearest center
        distances = np.array([min(np.linalg.norm(x - c) for c in centers) ** 2 for x in X])
        # Choose the next center with probability proportional to squared distance
        probabilities = distances / np.sum(distances)
        new_center_index = np.random.choice(len(X), p=probabilities)
        new_center = X[new_center_index]
        # Add the new center to the list of centers
        centers.append(new_center)
    
    return np.array(centers)

def three_center_cost(X, centers):
    max_distance = max(np.linalg.norm(x - centers[np.argmin([np.linalg.norm(x - c) for c in centers])]) for x in X)
    return max_distance

def three_means_cost(X, centers):
    total_distance = sum(np.linalg.norm(x - centers[np.argmin([np.linalg.norm(x - c) for c in centers])]) ** 2 for x in X)
    return total_distance / len(X)

def lloyds(X, initial_subset, max_iters=100):
    centers = initial_subset
    
    for _ in range(max_iters):
        # Assign each point to the nearest center
        clusters = [[] for _ in range(len(centers))]
        for x in X:
            distances = [np.linalg.norm(x - c) for c in centers]
            nearest_center_index = np.argmin(distances)
            clusters[nearest_center_index].append(x)
        
        # Update cluster centers
        new_centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        
        # Check for convergence
        if np.allclose(centers, new_centers):
            break
        
        centers = new_centers
    
    return centers, clusters

def ProblemOne():
    print("\nProblem One:")
    K = 3
    gonzal = gonzalez_algorithm(Y,K)
    kmean = kmeans_plusplus(Y,K)
    print('Gonzalez Center: ', three_center_cost(Y,gonzal))
    print('Gonzalez Mean: ', three_means_cost(Y,gonzal))
    print('Kmean++ Center: ', three_center_cost(Y,kmean))
    print('Kmean++ Mean: ', three_means_cost(Y,kmean))

    plt.figure(figsize=(10, 6))
    plt.scatter(Y[:, 0], Y[:, 1], color='blue', label='Dataset')
    plt.scatter(gonzal[:, 0], gonzal[:, 1], color='red', marker='x', label='Gonzalez Centers')
    plt.scatter(kmean[:, 0], kmean[:, 1], color='yellow', marker='o', label='k-Means++ Centers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cluster Centers')
    plt.legend()
    plt.grid(True)
    plt.show()


def ProblemTwo():
    print("\nProblem 2:")
    initial_subset = []
    initial_subset.append(Y[0:3])
    
    lloyd_center, lloyd_cluster = lloyds(Y,initial_subset,max_iters=100)
    means = three_means_cost(lloyd_cluster,lloyd_center)
    print('Lloyd Center: ',lloyd_center)
    print('Lloyd Mean: ', means)


def ProblemThree():
    print("\nProblem 3:")
    initial = []
    initial.append(gonzalez_algorithm(Y,3))
    lloyd_center, lloyd_cluster = lloyds(Y,initial,max_iters=100)
    means = three_means_cost(lloyd_cluster,lloyd_center)
    print('Lloyd Center: ',lloyd_center)
    print('Lloyd Mean: ', means)

def ProblemFour():
    print("\nProblem 4:")
    initial = []
    for _ in range(20):
    # Run k-Means++ to initialize centers
        initial_centers = kmeans_plusplus(Y, 3)
    
    # Run Lloyd's Algorithm with the initial centers from k-Means++
        final_center, final_cluster = lloyds(Y, initial_centers)
        
    
    # Compute the 3-means cost for this trial
        cost = np.mean(np.min(np.linalg.norm((Y[:, np.newaxis] - final_center), axis=2), axis=1) ** 2)        
        initial.append(cost)

    # Compute the average 3-means cost over 20 trials
        average_cost = np.mean(initial)

    print("Average 3 means cost over 20 trials", average_cost)


ProblemOne()
ProblemTwo()
ProblemThree()
ProblemFour()










    



