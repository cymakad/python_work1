# -*- coding: utf-8 -*-
# This work is done by Mak Cheuk Yin (20462137)
# The DBScan functions for assignment 3 is here

# Import necessary packages
import numpy as np
import pandas as pd
from time import time
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Load the dataset as pandas DataFrame
# Same as the one in a3_cymakad_k_means.py
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

# Select the neighbors for each data point
# And check whether their distances are smaller or equal to Epsilon
def find_all_neighbors(df, Epsilon, metric='euclidean'):
    '''Select all neighbors for each data point by calculating their
    point-to-point distance and check whether their distances are smaller
    or equal to Epsilon.
    
    df(pandas DataFrame): The dataFrame you want to pick data points\t
    Epsilon(int/float): The maximum distance between two\t
    metric(str, optional): The metric for calculating the distance 
    between two points \t
    --- default as 'euclidean'\t'''
    all_neighbors = {}
    all_distance = cdist(df.values, df.values, metric=metric)
    all_result = np.less_equal(all_distance, Epsilon)
    for index, result in enumerate(all_result):
        neighbors = np.argwhere(result).flatten()
        all_neighbors[index] = neighbors
    return all_neighbors

# Find the core points in the Data
def find_core_points(all_neighbors, MinPoints):
    '''Find all core points whose have at least MinPoints neighbors.
    
    all_neighbors(dict of array): The dictionary which records all neighbors
    for each data point\t
    MinPoints(int): The mimimum no. of data points for a core point'''
    core_points = np.array([])
    for index, neighbors in all_neighbors.items():
        if len(neighbors) >= MinPoints:
            core_points = np.concatenate((core_points, [index]))
    core_points = core_points.astype('int64')
    return core_points

# Find the border points in the Data
def find_border_points(all_neighbors, MinPoints, core_points):
    '''Find all border points whose have not enough neighbors (smaller than MinPoints)
    but the neighbors include any one of core points.
    
    all_neighbors(dict of array): The dictionary which records all neighbors
    for each data point\t
    MinPoints(int): The mimimum no. of data points for a core point\t
    core_points(array): The core points whose have at least MinPoints neighbors'''
    border_points = np.array([])
    for index, neighbors in all_neighbors.items():
        if len(neighbors) < MinPoints and np.isin(neighbors, core_points).any():
            border_points = np.concatenate((border_points, [index]))
    border_points = border_points.astype('int64')
    return border_points

# Find the noise points in the Data
def find_noise_points(df, core_points, border_points):
    '''Find all noise points whose are not in core points and border points.
    
    df(pandas DataFrame): The dataFrame you want to pick data points\t
    core_points(array): The core points whose have at least MinPoints neighbors\t
    border_points(array): The border points whose have core point neighbors'''
    all_points = df.index
    non_noise_points = np.concatenate((core_points, border_points))
    noise_points = np.setdiff1d(all_points, non_noise_points)
    return noise_points

# Merge clusters
def merge_clusters(clusters_array):
    '''Merge the existed clusters into the more general clusters if
    the two or more existed clusters have common neighbors.
    
    clusters_array(list of array): The list of all neighbors / points in each cluster'''
    new_clusters_array = []
    # Loop all cluster neighbors over each cluster array
    for index in range(len(clusters_array)):
        clus_neighbors = clusters_array[index]
        # Set the rule of whether merge two clusters into one
        selection = []
        # Loop over all new merged clusters,
        # Find whether the last merged and new merged cluster have common neighbor
        for clus in new_clusters_array:
            selection.append(np.isin(clus, clus_neighbors).any())
        # If there is any one common neighbor, merge them
        if np.any(selection):
            # Loop over all new merged clusters,
            # If the particular pair have common neighbor, merge them and save
            for index, clus in enumerate(new_clusters_array):
                if selection[index]:
                    combined = np.concatenate((clus, clus_neighbors))
                    new_clus = np.unique(combined)
                    new_clusters_array[index] = new_clus
        # If there is no common neighbor, just append it for the next search
        else:
            new_clusters_array.append(clus_neighbors)
    return new_clusters_array

# Generate dictionary for all clusters
def gen_clus_dict(clusters_array):
    '''Generate clusters dictionary from the completely merged clusters.
    
    clusters_array(list of array): The list of all neighbors / points in each cluster'''
    clus_dict = {}
    for i, cluster in enumerate(clusters_array):
        clus_name = 'Cluster_{}'.format(i+1)
        clus_dict[clus_name] = cluster
    return clus_dict

# Class DBScan
class DBScan():
    # Set the initial parameters for DBScan
    def __init__(self, Epsilon, MinPoints, **kwargs):
        '''initialize the DBScan class.
        
        Epsilon(int/float): The maximum distance between two\t
        MinPoints(int): The mimimum no. of data points for a core point\t
        inputfile(str, optional): The file name you want to load\t
        --- default as 'a3dataset.txt'\t
        columns(list of str, optional): The columns you want to put
        --- default as ['x1', 'x2']\t'''
        # Record the starting time
        self.starttime = time()
        # Load the data
        # The file name and columns can be changed
        self.df = data_to_pd_df(**kwargs)
        # Record the Epsilon and MinPoints
        self.Epsilon = Epsilon
        self.MinPoints = MinPoints
        return None
    
    def fit_and_cluster(self, metric='euclidean'):
        '''Scan through all data points in dataframe and cluster them using
        density-based apporach with Epsilon and MinPoints.
        
        metric(str, optional): The metric for calculating the distance
        between two points \t
        --- default as 'euclidean'\t'''
        # Select the neighbors for each data point
        # And check whether their distances are smaller or equal to Epsilon
        self.all_neighbors = find_all_neighbors(self.df, self.Epsilon, 
                                           metric='euclidean')
        # Identify the core points, border points and noise points in the data
        self.core_points = find_core_points(self.all_neighbors, self.MinPoints)
        self.border_points = find_border_points(self.all_neighbors, self.MinPoints, 
                                                self.core_points)
        self.noise_points = find_noise_points(self.df, self.core_points, 
                                              self.border_points)
        # Find all clusters in terms of core points
        all_neighbors_array = list(self.all_neighbors.values())
        clusters_array = np.take(all_neighbors_array, self.core_points).tolist()
        last_clusters_array = []
        # Merge all clusters_array and confirmed by while loop
        while not clusters_array == last_clusters_array:
            last_clusters_array = clusters_array.copy()
            clusters_array = merge_clusters(last_clusters_array)
        # Generate dictionary for all clusters
        self.clusters_dict = gen_clus_dict(clusters_array)
        return self.clusters_dict
    
    # Give a brief conclusion of this clustering model
    def conclusion(self):
        '''Give a brief conclusion of this clustering model'''
        print('There are {} clusters.'.format(len(self.clusters_dict)))
        total_data_points = 0
        for clus_name, data in self.clusters_dict.items():
            data_points = len(data)
            print('For {}, there are {} data points.'.format(clus_name, data_points))
            total_data_points += data_points
        print('In total, there are {} data points and {} outlier.'.\
              format(total_data_points, len(self.noise_points)))
        print('The running time is {}s.\n'.format(time()-self.starttime))
        
    # Plot the all clusters
    def plot_it(self):
        '''plot all clusters with different color and label them'''
        outliers = self.df.iloc[self.noise_points]
        ax = plt.scatter(outliers.iloc[:,0], outliers.iloc[:,1], 
                         c='k', s=3, label='outliers')
        for clus_name, clus_neighbors in self.clusters_dict.items():
            data_points = self.df.iloc[clus_neighbors]
            ax = plt.scatter(data_points.iloc[:, 0], data_points.iloc[:, 1], 
                             label=clus_name)
        ax = plt.xlabel(self.df.columns[0])
        ax = plt.ylabel(self.df.columns[1])
        ax = plt.title('The DBScan with Epsilon = {} and MinPoints = {}'.\
                       format(self.Epsilon, self.MinPoints))
        return ax

# Run DBScan with the parameters (Epsilon, MinPoints): (5,10);(5,4);(1,4)
# Plot them and save them
first = DBScan(5,10)
f_clus = first.fit_and_cluster()
first.conclusion()
fig = first.plot_it()
plt.savefig('first_scan.jpg')
plt.clf()

second = DBScan(5,4)
s_clus = second.fit_and_cluster()
second.conclusion()
fig = second.plot_it()
plt.savefig('second_scan.jpg')
plt.clf()

third = DBScan(1,4)
t_clus = third.fit_and_cluster()
third.conclusion()
fig = third.plot_it()
plt.savefig('third_scan.jpg')
plt.clf()
