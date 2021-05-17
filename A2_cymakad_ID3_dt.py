# -*- coding: utf-8 -*-
# This work is done by Mak Cheuk Yin (20462137)
# Here is a decision tree code with ID3 method

# Import the necessary packages
from A2_cymakad_decision_tree import *
from scipy.stats import entropy
import numpy as np
from time import time

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

# Run ID3 dicision tree and time it
start_time = time()
ID3_dt = decision_tree()
paths, tree = ID3_dt.fit(ID3_info)
predictions = ID3_dt.predict()
conclusion(start_time, predictions)