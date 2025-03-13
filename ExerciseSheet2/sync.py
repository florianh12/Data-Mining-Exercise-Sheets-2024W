import numpy as np
from typing import Tuple

class SynC:
    """Implements the SynC Algoritm

    Attributes:
        epsilon (float): the maximum distance (vector norm) for points to be in the same neighborhood.
        lambda_val (float): The method cutoff threshold that measures how close the points witin the neigborhood are <= 1.0.
        min_points (int): The amount of points in the neigborhood necessary to change a point from an outlier to a cluster.
        clustering_precision (int): The amoubnt of decimals taken into account when clustering synchronized points at the same location. 

    Methods:
        fit_predict(X:numpy.ndarray): Returns clustering, outliers and synchronized points in a Tuple.
    """
    epsilon: float
    lambda_val: float
    min_points: int
    clustering_precision: int
    labels_: int
    Ds_: np.ndarray

    def __init__(self, epsilon:float = 1.0, lambda_val:float = 1-1e-3, min_points:int = 5, clustering_precision = 3):
        self.epsilon = epsilon
        self.lambda_val = lambda_val
        self.min_points = min_points
        self.clustering_precision = clustering_precision

    def _compute_neighborhood(self, D:np.ndarray, p:np.ndarray) -> np.ndarray:
        """Computes neighborhood of a point in a dataset. 
        
        Args:
            D (numpy.ndarray): The original dataset.
            p (numpy.ndarray): The point used for neighborhood calculation. 

        Returns:
            numpy.ndarray: The Neighborhood of point p.
        """
        return D[np.linalg.norm(D-p, axis=1) <= self.epsilon]

    # N in the paper refers to Nb.shape[0]
    def _compute_r_p(self, Nb:np.ndarray, p:np.ndarray) -> float:
        """Computes r for a point in a neighborhood. 
        
        Args:
            Nb (numpy.ndarray): The neighborhood.
            p (numpy.ndarray): The point r is calculated for. 

        Returns:
            numpy.ndarray: The rp value for point p.
        """
        return np.sum(np.exp(-np.linalg.norm(Nb-p, axis=1)))/Nb.shape[0]
    
    def _cluster_data(self, Ds:np.ndarray):
        """Clusters given data into outliers and clusters.
        
        Args:
            D (numpy.ndarray): The original dataset
            Ds (numpy.ndarray): The synchronized data from the fit_predict method
        """
        # prepare np.ndarray with the dataset size 
        self.labels_ = np.zeros(Ds.shape[0],dtype=int)

        # accuracy for cluster detection, to reduce noise that shouldn't be noise 
        rounded_points = np.round(Ds,decimals=self.clustering_precision)

        # get distinct points in Ds and their counts
        unique_points, counts = np.unique(rounded_points,axis=0,return_counts=True)
        # turn the information into a dictionary
        clust_dict = dict(zip(map(tuple,unique_points), counts))

        # assign a clustering to the entry if count >= min_count else -1 (noise)
        current_cluster = 0
        for point in unique_points:
            point_tuple = tuple(point)

            if clust_dict[point_tuple] >= self.min_points:
                clust_dict[point_tuple] = current_cluster
                current_cluster += 1
            else:
                clust_dict[point_tuple] = -1

        # assign the clustering according to the dictionary
        for index, point in enumerate(rounded_points):
            self.labels_[index] = clust_dict[tuple(point)]
    
    def fit(self,X:np.ndarray):
        """Generates clustering for given data.

        Args:
            X (numpy.ndarray): The given dataset.
        """
        # rename dataset to fit paper pseudocode
        D = X        
        Ds = np.copy(D)        
        r_local:float = 0.0

        # loop until precision level has been reached
        while r_local <= self.lambda_val:
            # needs to reset for each loop
            r_local = 0.0

            for p in Ds:
                # calculate neighborhoo for synchronisation purposes
                Nb = self._compute_neighborhood(Ds,p)

                # calculate new p (x(t+1))
                p += (1/Nb.shape[0])*np.sum(np.sin(Nb - p),axis=0)
                
            # calculate r_local seperaly from p to get the updated rp values
            for p in Ds:
                # recompute neighborhood for accurate r_local results
                Nb = self._compute_neighborhood(Ds,p)

                # calculate r value for p and add it to r_local
                r_local += self._compute_r_p(Nb,p)
                
            # normalize r_local
            r_local /= D.shape[0]

            # raise error if there has been a miscalculation
            if r_local > 1:
                raise Exception("r_local too big")
        
        # save latest Ds 
        self.Ds_ = Ds

        self._cluster_data(Ds)
    
    def fit_predict(self, X:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generates clustering for given data and returns it.

        Args:
            X (numpy.ndarray): The given dataset.

        Returns:
            Tuple[ numpy.ndarray, numpy.ndarray]: clustering and outliers/noise (-1), synchronized data.
        """
        self.fit(X)

        return (np.copy(self.labels_),np.copy(self.Ds_))
