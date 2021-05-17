# -*- coding: utf-8 -*-
# This work is done by Mak Cheuk Yin (20462137)
# The k_means functions for assignment 3 is here

# Import necessary packages
import numpy as np
import pandas as pd
from time import time
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Load the dataset as pandas DataFrame
def data_to_pd_df(inputfile='a3dataset.txt', columns=['x1', 'x2']):
    '''Load the dataset with the name of a3dataset.txt with the columns
    named x1,x2 (defualt).
    
    inputfile(str, optional): The file name you want to load\t
    --- default as 'a3dataset.txt'\t
    columns(list of str, optional): The columns you want to put
    --- default as ['x1', 'x2']\t'''
    df = pd.read_csv(inputfile, header=None)
    df.columns = columns
    return df

# Randomly picking the k data points from df
def pick_data(df, k, random_seed=41891021):
    '''Pick the k data points from a pandas dataFrame (df) with 
    a random_seed which defaults as 41891021
    
    df(pandas DataFrame): The dataFrame you want to pick data points\t
    k(int): The no. of data points you want to pick\t
    random_seed(int, optional): The value for random calculation
    --- default as 41891021'''
    data_points = df.sample(n=k, random_state=random_seed)
    data_points = np.array(data_points)
    return data_points

# Cluster all data points into all k-means center/initial guessing
def clustering_data(df, center, metric='euclidean'):
    '''Clustering the data in the DataFrame into different centers using
    a specified metric which defaults as euclidean
    
    df(pandas DataFrame): The dataFrame you want to pick data points\t
    center(list of array): The center points for cllustering\t
    metric(str, optional): The metric for calculating the distance \
    between two points \t
    --- default as 'euclidean'\t'''
    # Find all distance and find the minimum one
    all_dist = cdist(df.values, center, metric=metric)
    min_dist = all_dist.argmin(axis=1)
    # Store the data points into clusters which have the minimum distance
    all_clusters_data = {}
    for Ci in range(len(center)):
        name = 'Cluster_{}'.format(Ci+1)
        clus_data_points = np.argwhere(min_dist == Ci)
        all_clusters_data[name] = clus_data_points.flatten()
    return all_clusters_data

# Update the centers by calculating the mean of clusters
def update_center(df, current_clusters_data):
    '''Calculating the mean of all data points for each cluster and set
    it as the new cluster center
    
    df(pandas DataFrame): The dataFrame you want to pick data points\t
    current_clusters_data(dict of array): All data points from current clusters set'''
    new_centers = []
    for Ci, data_points in current_clusters_data.items():
        data_avg = df.iloc[data_points].mean()
        new_centers.append(np.array(data_avg))
    new_centers = np.array(new_centers)
    return new_centers

# Check whether the two sets of all clusters data are the same
def is_same_clusters(last_clusters_data, current_clusters_data):
    '''Check whether the two sets of all clusters data are the same
    
    last_clusters_data(dict of array): All data points from last clusters set\t
    current_clusters_data(dict of array): All data points from current clusters set'''
    # If they have same keys, then continue otherwise return False 
    if last_clusters_data.keys() == current_clusters_data.keys():
        each_result = []
        # Loop over all cluster to check whether their data points are the same
        for key in last_clusters_data.keys():
            result = np.array_equal(last_clusters_data[key], 
                                    current_clusters_data[key])
            each_result.append(result)
        return all(each_result)
    else:
        return False

# Class of K_Means
class k_means():
    # Set the initial parameters for k_means
    def __init__(self, k, **kwargs):
        '''initialize the k_means class
        
        k(int): the no. of clusters we want\t
        inputfile(str, optional): The file name you want to load\t
        --- default as 'a3dataset.txt'\t
        columns(list of str, optional): The columns you want to put
        --- default as ['x1', 'x2']\t'''
        # Record the starting time
        self.starttime = time()
        # Load the data
        # The file name and columns can be changed
        self.df = data_to_pd_df(**kwargs)
        self.k = k # Save the k (clustering number)
        return None
    
    # Fit the model and optimize the clustering
    def fit_and_cluster(self, random_seed=41891021, metric='euclidean'):
        '''fit the model by the DataFrame and cluster it into k clusters
        
        random_seed(int, optional): The value for random calculation
        --- default as 41891021\t
        metric(str, optional): The metric for calculating the distance \
        between two points \t
        --- default as 'euclidean'\t'''
        # Find the initial guess
        # The random seed can be changed
        initial_guess = pick_data(self.df, self.k, random_seed)
        # Find all clusters data points from initial guess
        # The metric can be changed
        current_clusters_data = clustering_data(self.df, initial_guess, 
                                                metric)
        last_clusters_data = {}
        # Loop to find new clusters until there are no change
        while not is_same_clusters(last_clusters_data, current_clusters_data):
            # Update centers
            new_centers = update_center(self.df, current_clusters_data)
            last_clusters_data = current_clusters_data.copy()
            current_clusters_data = clustering_data(self.df, new_centers,
                                                    metric)
        self.cluster_center = new_centers
        self.clusters = current_clusters_data
        return current_clusters_data
    
    # Calculate the squared error for each cluster
    def cal_SE(self, metric='euclidean'):
        '''Calculate the squared error for each clusters
        
        metric(str, optional): The metric for calculating the distance \
        between two points \t
        --- default as 'euclidean'\t'''
        all_squared_error = {}
        for Ci, name in enumerate(self.clusters):
            data_center = [self.cluster_center[Ci]]
            data_points = self.clusters[name]
            data_points = self.df.iloc[data_points].values
            clus_dist = cdist(data_points, data_center, metric=metric)
            squared_error = np.sum(clus_dist**2)
            all_squared_error[name] = squared_error
        self.all_squared_error = all_squared_error
        return all_squared_error
            
    # Give a brief conclusion of this clustering model
    def conclusion(self):
        '''Give a brief conclusion of this clustering model'''
        print('There are {} clusters.'.format(len(self.cluster_center)))
        total_data_points = 0
        all_squared_error = self.cal_SE()
        for clus_name, data in self.clusters.items():
            data_points = len(data)
            print('For {}, there are {} data points and the squared error is {}.'.\
                  format(clus_name, data_points, all_squared_error[clus_name]))
            total_data_points += data_points
        print('In total, there are {} data points and the total squared error is {}.'.\
              format(total_data_points, sum(all_squared_error.values())))
        print('The running time is {}s.\n'.format(time()-self.starttime))
        
    # Plot the all clusters
    def plot_it(self):
        '''plot all clusters with different color and label them'''
        for Ci, clus_name in enumerate(self.clusters):
            center = self.cluster_center[Ci]
            data_points = self.clusters[clus_name]
            data_points = self.df.iloc[data_points]
            ax = plt.scatter(data_points.iloc[:, 0], data_points.iloc[:, 1], 
                             label=clus_name)
            ax = plt.scatter(center[0], center[1], c='blue')
            ax = plt.annotate(clus_name, xy=center, alpha=0.7, c='k')
        ax = plt.scatter(center[0], center[1],c='blue', label='centers')
        ax = plt.xlabel(self.df.columns[0])
        ax = plt.ylabel(self.df.columns[1])
        ax = plt.legend()
        ax = plt.xlim(0,900)
        ax = plt.title('The K-Means Clustering with k = {}'.format(self.k))
        return ax

# Run k_means with k = 3, 6, 9 and plot them
three = k_means(3)
three_clusters = three.fit_and_cluster()
three.conclusion()
fig = three.plot_it()
plt.savefig('3_means.jpg')
plt.clf()

six = k_means(6)
six_clusters = six.fit_and_cluster()
six.conclusion()
fig = six.plot_it()
plt.savefig('6_means.jpg')
plt.clf()

nine = k_means(9)
nine_clusters = nine.fit_and_cluster()
nine.conclusion()
fig = nine.plot_it()
plt.savefig('9_means.jpg')
plt.clf()
    
    