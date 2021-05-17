# -*- coding: utf-8 -*-
# This work is done by Mak Cheuk Yin (20462137)
# Here is a decision tree code with C4.5 method
# As C4.5 is based on ID3, the ID3 will be imported
# The only difference is the split info used in C4.5 but not in ID3
# As the split info can reduce the bais from a large number of values

# Import the necessary packages
from A2_cymakad_decision_tree import *
from scipy.stats import entropy
import numpy as np
from time import time

def C4dot5_info(AVC_set, label_info):
    # calculate the gain ratio with C4.5 method for one AVC set
    # Call back the ID3 information gain for further calculation
    ID3_gain = ID3_info(AVC_set, label_info)
    # Find the ratio of all values in an attribute
    values_ratio = AVC_set.sum(axis=1)/AVC_set.sum().sum()
    # Calculate the split info with entropy using the values_ratio
    split_info = entropy(values_ratio, base=2)
    # The information gain from ID3 divided by the split info to find gain ratio 
    gain_ratio = ID3_gain/split_info
    return gain_ratio

# Run C4.5 decision tree and time it
start_time = time()
C4dot5_dt = decision_tree()
paths, tree = C4dot5_dt.fit(C4dot5_info)
predictions = C4dot5_dt.predict()
conclusion(start_time, predictions)
    