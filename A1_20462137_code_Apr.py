# -*- coding: utf-8 -*-
'''
The apriori is done by Mak Cheuk Yin under Python 3.7
'''

# Import the necessary package
import numpy as np
from time import time
from collections import Counter
from itertools import combinations, repeat

# Minimun Support we needed
min_sup = 400

# Sort the data and return a 2D list
# Input can be 0D, 1D, 2D
def sort_it(data_list):
    sort_int_list = sorted([sorted([*map(int, data)]) for data in data_list])
    sort_data_list = [[*map(str, int_data)] for int_data in sort_int_list]
    return sort_data_list

# load the data
def load_data(file):
    with open(file, 'r') as a1:
        data_list = [line.strip().rsplit(' ') for line in a1]
    return data_list

def flattern_data_unique(data_list):
    # Flatten the data and find the unique data
    array_data = [np.array(list(item)) for item in data_list]
    flat_data = np.unique(np.concatenate(array_data, axis=None)).tolist()
    return flat_data

def initialize(data_list):
    # Use Counter method find 1-item frquent itemsets
    array_data = [np.array(data) for data in data_list]  # Flatten the data
    flat_data = np.concatenate(array_data, axis=None).tolist()
    first_count = Counter(flat_data)
    one_itemsets = [list([list(first_count.keys())[i]]) 
                    for i, v in enumerate(first_count.values())
                    if v >= min_sup]
    return one_itemsets

def form_combination(itemsets, k):
    # Form new combinations from itemsets
    frequent_items = flattern_data_unique(itemsets)
    combin_list = [list(itemset) for itemset in combinations(frequent_items, k)]
    combin_list = sort_it(combin_list)
    return frequent_items, combin_list

def infrequent_combination(old_combin_list, new_itemsets):
    # Find the infrequent k-1-item combinations
    infrequent = [old_combin 
               for old_combin in old_combin_list 
               if not old_combin in new_itemsets]
    return infrequent

def issuperset(combin, itemset):
    # Check a fixed combination with the itemset
    if frozenset(combin).issuperset(frozenset(itemset)):
        return True
    else:
        return False

def prune_combination(new_combin_list, infrequent, new_itemsets, k):
    # prune combination with two steps to speed up
    # 1st. Check the k-item combinations with the k-1-item new_itemsets
    selected_combin = []
    for new_combin in new_combin_list:
        if any(list(map(issuperset, repeat(new_combin), new_itemsets))):
            selected_combin.append(new_combin)
    # 2nd. Check the k-items combinations with the k-1-items infrequent combinations
    # To reduce the no. of loop, use back combinations(k-1)
    needed_combin = []
    for combin in selected_combin:
        if not any([list(sub_combin) in infrequent
                    for sub_combin in combinations(combin, k-1)]):
            needed_combin.append(combin)
    return needed_combin

def is_frequent_itemset(data_list, frequent_items,\
                        new_combin_list, min_sup, k):
    # Build dict for new_combinations_list
    new_combin_dict = {}
    for new_combin in new_combin_list:
        new_combin_dict[tuple(new_combin)] = 0
    # Check the transcations with the items of all combinations
    # from frequent itemsets
    frequent_items = sorted([*map(int, frequent_items)])
    frequent_items = [str(item) for item in frequent_items]
    for items in data_list:
        transcation = items.copy()
        for item in items:
            if not item in frequent_items:
                transcation.remove(item)
        # Check the remaining items in transcation
        # And then count the combinations
        if len(transcation) >= k:
            transcation_combinations = [*map(list, combinations(transcation, k))]
            if k > 2: # Withdraw unnecessary combinations after 2nd run
                transcation_combinations = [combin 
                                    for combin in transcation_combinations
                                     if combin in new_combin_list]
            for combin in transcation_combinations:
                if tuple(combin) in new_combin_dict:
                    new_combin_dict[tuple(combin)] += 1
    # Check the value of the bucket which is larget than min_sup or not
    new_itemsets = []
    for tuple_combin in new_combin_dict:
        if new_combin_dict[tuple_combin] >= min_sup:
            new_itemsets.append(list(tuple_combin))
    return new_itemsets

def Apriori(file):
    # Find all frequent itemsets with Apriori Algorithm
    starttime = time()
    data_list = load_data(file)
    new_itemsets = initialize(data_list)
    frequent_itemsets = new_itemsets
    k = 2
    while new_itemsets:
        frequent_items, new_combin_list = form_combination(new_itemsets, k)
        if k > 2:
            infrequent = infrequent_combination(old_combin_list, new_itemsets)
            needed_combin_list = \
            prune_combination(new_combin_list, infrequent, new_itemsets, k)
            new_itemsets = is_frequent_itemset(data_list, frequent_items,\
                                               needed_combin_list, min_sup, k)
        else:
            new_itemsets = is_frequent_itemset(data_list, frequent_items,\
                                               new_combin_list, min_sup, k)
        old_combin_list = new_combin_list
        frequent_itemsets += new_itemsets
        k += 1
    return frequent_itemsets, starttime
    
# Run Apriori and time it 
frequent_itemsets, starttime = Apriori('a1dataset.txt')
# Print the result and running time
print('Running Time: {}s\nFrequent itemsets are {}:\n{}'.format(
        (time() - starttime), len(frequent_itemsets), frequent_itemsets))
