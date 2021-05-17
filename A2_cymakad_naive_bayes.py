# -*- coding: utf-8 -*-
# This work is done by Mak Cheuk Yin (20462137)
# The naive bayes classifier for assignment 2 is here

# Import the necessary packages
from A2_cymakad_decision_tree import *
from scipy.stats import entropy
import numpy as np
import pandas as pd
from time import time

class naive_bayes_classifier():
    def __init__(self):
        # Initilize the classifier by the training and testing data
        # Read the training and testing data
        self.X_train, self.y_train, self.X_test = get_df_from_data()
        # Find all values in training data
        self.all_train_values = find_unique_train_values(self.X_train, 
                                                         self.y_train)
        self.possible_class = ['not_recom', 'recommend', 'very_recom',\
                               'priority', 'spec_prior']
        return None
    
    def find_all_priors(self):
        # Find all prior from the training data (y)
        all_priors = {}
        # Count the possible values in y
        y_train_sum = self.y_train.value_counts()
        # Ensure the target have all possible classes
        missing_class = str(*set(self.possible_class).difference(
            set(y_train_sum.index)))
        y_train_sum[missing_class] = 0
        # Laplace Correction
        corrected_sum = y_train_sum + 0.0001
        # Turn the count sum to probability
        prob = corrected_sum/corrected_sum.sum()
        # Save the name of target
        target = prob.name
        # Save all prior calculated class by class
        for t_class in prob.index:
            label = (target, t_class)
            all_priors[label] = prob[t_class]
        return all_priors
    
    def find_all_likehoods(self, all_AVC_sets):
        # Find all likehoods from the training data (X) with help of AVC sets
        all_likehoods = {}
        # Loop all attr in all AVC sets
        for attr in all_AVC_sets.keys():
            AVC_set = all_AVC_sets[attr]
            # Ensure the target have all possible classes
            missing_class = str(*set(self.possible_class).difference(
                set(AVC_set.columns)))
            AVC_set[missing_class] = 0
            # Laplace Correction
            corrected_set = AVC_set + 0.0001
            # Save the name of target
            target = corrected_set.columns.name
            # Loop all classes of target
            for t_class in corrected_set.columns:
                # Save all attribute values in one class
                attr_values = corrected_set[t_class]
                # Trun the attribute values to probability
                prob = attr_values/attr_values.sum()
                # Loop all attribute values
                for value in prob.index:
                    # Save all likehoods with the label
                    # label means that 'attr = value | target = t_class'
                    label = (attr, value, target, t_class)
                    all_likehoods[label] = prob[value]
        return all_likehoods
    
    def fit(self):
        # Fit the naive bayes classifier with training data
        # Concatenate X_train and y_train
        df = pd.concat([self.X_train, self.y_train], axis = 1)
        # Build all AVC sets with df
        all_AVC_sets = build_AVC_set(df)
        # Find all priors and save them
        self.all_priors = self.find_all_priors()
        # Find all likehoods and save them
        self.all_likehoods = self.find_all_likehoods(all_AVC_sets)
        return self.all_priors, self.all_likehoods
    
    def predict(self):
        # Predict the target class (y) for the testing data (X_test)
        predictions = []
        # Save the target name
        target = self.y_train.name
        # Loop all data array in X_test
        for data_no in range(len(self.X_test)):
            # Find the posterioris for one data array
            # The posterioris calculated are just the likehood * prior
            # We can ignore the part of evidence as they are constant in
            # each data point / transaction
            # We can compare the simplified one to determine the prediction
            posterioris = {}
            test_data = self.X_test.iloc[data_no, :]
            # Loop the class label from all possible class label
            for t_class in self.possible_class:
                # Find the posteriori for one class label
                # Extract the prior of this class label
                prior = self.all_priors[(target, t_class)]
                # Find the likehood of this data array for this class
                # Set the likehood to 1 for further multiplication
                data_likehood = 1
                # Loop all attributes in this data array
                for attr in test_data.index:
                    # Find the corresponding value of one attribute
                    value = test_data[attr]
                    # Set the key as label for referring all_likehoods
                    label = (attr, value, target, t_class)
                    # Extract the value in all_likehood and 
                    # multiply back to the data_likehood
                    data_likehood *= self.all_likehoods[label]
                # After calculating the likehood of all attributes in data,
                # posteriori of this class can be calculated by the 
                # multiplication of data_likehood and prior of this class
                posterioris[t_class] = data_likehood * prior
            # After calculating all posterioris of all classes for one data
            # array, the prediction will be the class of the largest posteriori
            pred_index = np.argmax(list(posterioris.values()))
            # Save all predictions
            predictions.append(list(posterioris.keys())[pred_index])
        # Return the predictions as a pandas series with the target name
        y_test_pred = pd.Series(predictions, name=self.y_train.name)
        return y_test_pred
    
    def fit_and_predict(self):
        # Fit and predict the naive bayes classifier with training and testing
        # data in one function
        # The fitting part
        all_priors, all_likehoods = self.fit()
        # The prediction part
        predictions = self.predict()
        # Return the predicitions
        return predictions

# Run the naive bayes classifier and time it
start_time = time()
naive_bc = naive_bayes_classifier()
predictions = naive_bc.fit_and_predict()
conclusion(start_time, predictions)