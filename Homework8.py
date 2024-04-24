import seaborn as sb
import pandas as pd
import numpy as np
fields = ['s1', 's2']

dX = pd.read_csv('C:/Users/benbo/CoolerCodeApp/Projects/Python/Ck.csv')

def gonzalez_algorithm(X, K):
    # Choose the first center randomly
    centers = [X[np.random.randint(len(X))]]
    
    # Loop until we have K centers
    while len(centers) < K:
        # For each point, find the distance to the nearest center
        distances = np.array([min(np.linalg.norm(x - c) for c in centers) for x in X])
        # Choose the point with the maximum distance as the next center
        new_center = X[np.argmax(distances)]
        # Add the new center to the list of centers
        centers.append(new_center)
    
    return centers

gonzalez_algorithm(dX,3)


