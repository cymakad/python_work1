# -*- coding: utf-8 -*-
# This work is done by Mak Cheuk Yin (20462137)
# The common functions for assignment 2 is here

# Import the necessary packages
import pandas as pd
import numpy as np
from scipy.stats import entropy
from itertools import repeat
from time import time

def get_df_from_data(train='train.txt', test='test.txt'):
    # Load the data to get the pandas dataframe 
    # and separate the label and attributes
    df_train = pd.read_csv(train, sep=',')
    X_train = df_train.iloc[:,:-1]
    y_train = df_train.iloc[:, -1]
    X_test = pd.read_csv(test, sep=',')
    return X_train, y_train, X_test

def find_unique_train_values(X_train, y_train):
    # Find all attributes and label values in the training dataset
    all_train_values = {}
    for i in range(len(X_train.columns)):
        column = X_train.columns[i]
        all_train_values[column] = list(X_train.iloc[:, i].drop_duplicates())
    all_train_values[y_train.name] = list(y_train.drop_duplicates())
    return all_train_values

def build_AVC_set(df):
    # Separate and Group the data into the AVC set
    label = df.columns[-1]
    column = df[label]
    all_AVC_set = {}
    for i in range(len(df.columns)-1):
        index = df.columns[i]
        AVC_set = df.pivot_table(columns=column, values=label, index=index, 
                                 aggfunc='count', fill_value=0)
        all_AVC_set[index] = AVC_set
    return all_AVC_set
    
def generate_lists(base, options):
    # generate multiple lists from base object with many options
    lists = [*zip(repeat(base), options)]
    return lists

def add_tuples_with_options(base, options):
    # add two tuples together and one can be multiple tuples (options)
    combine = lambda t1, t2: tuple(list(t1) + list(t2))
    lists = [*map(combine, repeat(base), options)]
    return lists

def conclusion(start_time, result):
    # Print the running time and result as conclusion
    print('The running time of this programme is {}s\nThe results are\n\
          '.format((time()-start_time)))
    print(result)

#==============================================================================
# Store it for using it back to C4.5_dt
def ID3_info(AVC_set, label_info):
    # calculate the information gain with ID3 method for one AVC set
    # Find the info of all values in this attribute
    values_info = [entropy(AVC_set.iloc[i, :], base=2)\
                  for i in range(len(AVC_set.index))]
    # Find the count ratio or weight of the values in this attribute
    values_ratio = AVC_set.sum(axis=1)/AVC_set.sum().sum()
    # Multiple the values_info and values_ratio to get the attribute info
    attr_info = np.multiply(values_info, values_ratio).sum()
    # Find the information gain of this attribute
    gain = label_info - attr_info 
    return gain

