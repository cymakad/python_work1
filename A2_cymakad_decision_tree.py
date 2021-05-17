# -*- coding: utf-8 -*-
# This work is done by Mak Cheuk Yin (20462137)
# Here is a general decision tree code

# Import necessary modules
from A2_cymakad_Common_function import *
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import entropy

def find_label_info(all_AVC_set):
    # Find the label info which is same in each round
    # Only expanding the tree can change the label info with different AVC_set
    label_info = entropy(list(all_AVC_set.values())[0].sum(), base=2)
    return label_info

def find_largest_gain_attr(all_AVC_set, label_info, method):
    # Find the largest gain attribute for the tree with a method
    # Find all attributes gain first
    all_attr_gain = {}
    for attr in all_AVC_set.keys():
        all_attr_gain[attr] = method(all_AVC_set[attr], label_info)
    largest_gain_index = np.argmax(list(all_attr_gain.values()))
    largest_gain_attr = list(all_attr_gain.keys())[largest_gain_index]
    return largest_gain_attr

class decision_tree():
    def __init__(self):
        # Initilize the tree by the training and testing data
        # Read the training and testing data
        self.X_train, self.y_train, self.X_test = get_df_from_data()
        # Find all values in training data
        self.all_train_values = find_unique_train_values(self.X_train, 
                                                         self.y_train)
        return None
    
    def make_the_decision(self, df, method):
        # Set the conditions
        # If the last column, which is target label, in the dataframe just
        # have one value, stop making new decision and return the only one 
        # value as result
        # If there is one column left, no more new decision can be generated,
        # the decision will stop and return the remaining options
        if len(df.iloc[:, -1].unique()) == 1 or len(df.columns) == 1:
            label_name = [df.iloc[:, -1].name]
            result_type = df.iloc[:, -1].unique()
            # All values will have a probaility of occurrences
            result_prob = list(df.iloc[:, -1].value_counts(
                            normalize=True))
            result = [*zip(result_type, result_prob)]
            # generate new paths
            paths = add_tuples_with_options(label_name, result)
        # If both conditions are false, the decision will continue
        else:
            all_AVC_set = build_AVC_set(df) # build AVC set for all features
            # Find the label(y) info for the same group of AVC_sets.
            # As they come from the same dataframe, label info is same
            label_info = find_label_info(all_AVC_set)
            # Find the attribute who have largest gain
            largest_gain_attr = find_largest_gain_attr(all_AVC_set, label_info, 
                                                       method=method)
            # generate new paths
            paths = generate_lists(largest_gain_attr,
                                   self.all_train_values[largest_gain_attr])
        return paths
    
    def find_incompleted_paths(self, paths):
        # Find the incompleted paths that are no target label (y)
        incompleted_paths = []
        for path in paths:
            if not self.y_train.name in path:
                incompleted_paths.append(path)
        return incompleted_paths
    
    def build_decision_tree(self, all_paths):
    # Build the tree from all_paths
        tree = lambda: defaultdict(tree)
        self.dt = tree()
        for path in all_paths:
            length = len(path)
            current_level = self.dt[path[0]]
            for i in range(1, length-2):
                current_level = current_level[path[i]]
            current_level[path[-2]] = path[-1]
        return None
    
    def fit(self, method):
        # Fit the decision tree with training data
        # Get the training dataframe
        df = pd.concat([self.X_train, self.y_train], axis=1)
        # The initial condition of making decision
        # Find the best decision
        paths = self.make_the_decision(df, method)
        # Find the incompleted paths that are no target label (y)
        paths = self.find_incompleted_paths(paths)
        all_paths = paths.copy() # Save it
        while paths: # Finish all paths by while loop
            for path in paths: # Loop all paths one by one
                current_df = df # result the dataframe
                # Select the essential data from dataframe and path we have
                for i in range(0, len(path), 2):
                    current_df = current_df[current_df[path[i]] == path[i+1]]
                    current_df = current_df.drop(columns=path[i])
                # Making decisions and finish the remaining path
                new_paths = self.make_the_decision(current_df, method)
                # Update the original paths
                new_paths = add_tuples_with_options(path, new_paths)
                all_paths.remove(path) # Remove the old and incompled path
                all_paths += new_paths # add the new paths into all_paths
            paths = self.find_incompleted_paths(all_paths)
        # Build the decision tree from all paths
        self.build_decision_tree(all_paths)
        return all_paths, self.dt
    
    def predict(self):
        # Predict the test data by the tree
        predictions = [] # Save the predictions
        for data_no in range(len(self.X_test)): # Loop all test data
            test_data = self.X_test.iloc[data_no, :]
            # initalize the parameters
            current_level = self.dt
            current_attr = str(*current_level.keys())
            # Using while loop to find out the tree path which matchs the
            # test data until the target label (y)
            while current_attr != self.y_train.name:
                current_value = test_data[current_attr]
                current_level = current_level[current_attr][current_value]
                current_attr = str(*current_level.keys())
            # If there is just one target values, take it as our predictions
            if len(current_level[current_attr].keys()) == 1:
                pred = str(*current_level[current_attr].keys())
                predictions.append(pred)
            # If not, return NA to further analysis
            else:
                predictions.append('NA')
        # Return the result as pandas series
        y_test_pred = pd.Series(predictions, name=self.y_train.name)
        return y_test_pred