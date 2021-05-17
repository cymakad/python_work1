# -*- coding: utf-8 -*-
'''
The FP growth is done by Mak Cheuk Yin under Python 3.7
'''

# Import the necessary package
import numpy as np
from time import time
from collections import Counter, defaultdict
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
    # Find the one-member frequent_itemsets and frequent items
    array_data = [np.array(data) for data in data_list]
    flat_data = np.concatenate(array_data, axis=None).tolist()
    first_count = Counter(flat_data)
    one_itemsets = [list([list(first_count.keys())[i]]) 
                    for i, v in enumerate(first_count.values())
                    if v >= min_sup]
    frequent_items = flattern_data_unique(one_itemsets)
    frequent_items = sorted([*map(int, frequent_items)])
    frequent_items = [*map(str, frequent_items)]
    return frequent_items

def frequent_transcations(data_list, frequent_items):
    # Select the transcations whose have the frequent item from one_itemsets
    frequent_data_list = []
    for transcation in data_list:
        needed_transcation = []
        for item in transcation:
            if item in frequent_items:
                needed_transcation.append(int(item))
        if needed_transcation: # Drop the blank transcation
            frequent_data_list.append(sorted(needed_transcation))
    sorted_frequent_data_list = []
    for transcation in sorted(frequent_data_list):
        sorted_transcation = [*map(str, transcation)]
        sorted_frequent_data_list.append(sorted_transcation)
    return sorted_frequent_data_list

def all_path_in_fp_tree(frequent_data_list):
    # Record all the path in fp tree
    all_paths = np.unique(frequent_data_list)
    all_paths = sort_it(all_paths)
    return all_paths
    
def build_fp_node(item, level):
    # Build the fp node and add 1 for the items passing through the tree
    tree = lambda: defaultdict(tree)
    if item in level:
        level[item][0] += 1
    else:
        level[item] = [1, tree()]
    current_level = level[item][1]
    return current_level

def build_fp_tree(frequent_transcations):
    # Build the fp tree from the frequent transcations
    tree = lambda: defaultdict(tree)
    fp_tree = tree()
    for transcation in frequent_transcations:
        k = len(transcation)
        current_level = build_fp_node(transcation[0], fp_tree)
        i = 1
        while not i == k:
            current_level = build_fp_node(transcation[i], current_level)
            i += 1
    return fp_tree

def paths_end_with_item(frequent_item, all_paths):
    # Select the path which ends with the frequent_item
    item_paths = [path for path in all_paths
                  if frequent_item in path]
    item_end_paths = [path[0 : path.index(frequent_item) + 1] 
                      for path in item_paths]
    item_end_paths = sort_it(np.unique(item_end_paths))
    return item_end_paths

def count_the_item_paths(item_end_paths, fp_tree):
    # Acess the tree with the end path to find the count of the path
    all_paths_count = {}
    for path in item_end_paths:
        current_level = fp_tree[path[0]]
        i = 1
        while not i == len(path):
            current_level = current_level[1][path[i]]
            i += 1
        all_paths_count[tuple(path)] = current_level[0]
    return all_paths_count
        
def frequent_items_in_paths(all_paths_count, min_sup):
    # Count each item in path to find out the frequent item in path
    items_count = {}
    for path in all_paths_count:
        for item in list(path):
            if item in items_count:
                items_count[item] += all_paths_count[path]
            else:
                items_count[item] = all_paths_count[path]
    frequent_items_count = {}
    for item in items_count:
        if items_count[item] >= min_sup:
            frequent_items_count[item] = items_count[item]
    return frequent_items_count

# If the length of frequent_items_count is 1,
# There is no need to do the followings
# as here cannot generate more frequent itemsets

def filter_path_with_frequent_items(all_paths_count, frequent_items_count):
    # Update the path and withdraw the infrequrn item in path
    # Then, Sum up the same updated path
    frequent_paths_count = {}
    for path in all_paths_count:
        frequent_path = tuple(np.intersect1d(list(path), 
                                       list(frequent_items_count.keys())))
        if frequent_path in frequent_paths_count:
            frequent_paths_count[frequent_path] += all_paths_count[path]
        else:
            frequent_paths_count[frequent_path] = all_paths_count[path]
    return frequent_paths_count

def add_back_the_superset_count(frequent_paths_count):
    # Last, add the count of superset into the count of subset
    keys = list(frequent_paths_count.keys())
    keys = sorted(keys, key=len)
    for i in range(len(keys)):
        for key in keys[:i+1]:
            if keys[i] != key and frozenset(keys[i]).issuperset(frozenset(key)):
                frequent_paths_count[key] += frequent_paths_count[keys[i]]
    return frequent_paths_count

# If the length of frequent_items_count is 2,
# There is no need to do the followings
# as here cannot generate more frequent itemsets

def find_frequent_itemsets(frequent_paths_count, frequent_item, min_sup):
    # Append the extra subsets of the paths whose have length longer than 2
    keys = list(frequent_paths_count.keys())
    for path in keys:
        if len(path) > 2:
            k = range(2, len(path))
            combin_subsets = [*map(combinations, repeat(path), k)]
            combin_subsets = [[*combin] for combin in combin_subsets]
            subsets = []
            for subset_list in combin_subsets:
                for subset in subset_list:
                    if frequent_item in subset:
                        subsets.append(subset)
            for subset in subsets:
                if not subset in keys:
                    if not subset in frequent_paths_count:
                        frequent_paths_count[subset] = 0
    frequent_paths_count = add_back_the_superset_count(frequent_paths_count)
    frequent_dict = {}
    # Using the path, to find the frequent_itemsets of that item
    for path in frequent_paths_count:
        if frequent_paths_count[path] >= min_sup:
            frequent_dict[path] = frequent_paths_count[path]
    return frequent_dict

def FP_Growth(file):
    # Find all frequent itemsets with FP Growth Algorithm
    starttime = time()
    data_list = load_data(file)
    frequent_items = initialize(data_list)
    frequent_data_list = frequent_transcations(data_list, frequent_items)
    fp_tree = build_fp_tree(frequent_data_list)
    all_paths = all_path_in_fp_tree(frequent_data_list)
    frequent_itemsets = {}
    for frequent_item in frequent_items:
        item_end_paths = paths_end_with_item(frequent_item, all_paths)
        all_paths_count = count_the_item_paths(item_end_paths, fp_tree)
        frequent_items_count = frequent_items_in_paths(all_paths_count, min_sup)
        if len(frequent_items_count) == 1: 
            # frequent_items_count at least is 1
            key = tuple(frequent_items_count.keys())
            frequent_itemsets[key] = frequent_items_count[frequent_item]
        elif len(frequent_items_count) == 2:
            frequent_paths_count = \
            filter_path_with_frequent_items(all_paths_count, frequent_items_count)
            frequent_paths_count = add_back_the_superset_count(frequent_paths_count)
            frequent_itemsets.update(frequent_paths_count)
        else:
            frequent_paths_count = \
            filter_path_with_frequent_items(all_paths_count, frequent_items_count)
            frequent_dict = find_frequent_itemsets(frequent_paths_count, frequent_item, min_sup)
            frequent_itemsets.update(frequent_dict)
    return frequent_itemsets, starttime

# Run FP_Growth and list out the frequent itemsets
frequent_itemsets, starttime = FP_Growth('a1dataset.txt')
frequent_itemsets_list = sort_it(list(frequent_itemsets.keys()))
frequent_itemsets_list = sorted(frequent_itemsets_list, key=len)
# Print the result and running time
print('Running Time: {}s\nFrequent itemsets are {}:\n{}'.format(
        (time() - starttime), len(frequent_itemsets_list), 
        frequent_itemsets_list))


