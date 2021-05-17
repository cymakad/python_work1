# -*- coding: utf-8 -*-
'''
The hash tree is done by Mak Cheuk Yin under Python 3.7
'''

# Import the necessary package
import numpy as np
from time import time
from collections import Counter, defaultdict
from itertools import combinations, repeat

# Minimun Support and the hash value (buckets) we needed
min_sup = 400
hash_value = 2216

# Sort the data and return a 2D list
# Input can be 0D, 1D, 2D
def sort_it(data_list):
    sort_int_list = sorted([sorted([*map(int, data)]) for data in data_list])
    sort_data_list = [[*map(str, int_data)] for int_data in sort_int_list]
    return sort_data_list

def hash(items):
    # The hash function we used
    return items % hash_value

# Hash the data
def hash_it(list_data):
    int_combin = sorted([*map(int, list_data)])
    hash_combin = [*map(hash, int_combin)]
    return list(hash_combin)

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
    array_data = [np.array(data) for data in data_list] # Flatten the data
    flat_data = np.concatenate(array_data, axis=None).tolist()
    first_count = Counter(flat_data)
    one_itemsets = [list([list(first_count.keys())[i]]) 
                    for i, v in enumerate(first_count.values())
                    if v >= min_sup]
    return one_itemsets


def form_combination(itemsets, k):
    # Form length-k combinations from k-1 itemsets 
    frequent_items = flattern_data_unique(itemsets)
    frequent_items = sorted([*map(int, frequent_items)])
    frequent_items = [*map(str, frequent_items)]
    combin_list = [list(itemset) for itemset in combinations(frequent_items, k)]
    hash_combin_list = [*map(hash_it, combin_list)]
    return frequent_items, hash_combin_list, combin_list
    
def infrequent_combination(old_combin_list, new_itemsets):
    # Find infrequent k combinations
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
    # 1st. Check the k combinations with the k-1 new_itemsets
    selected_combin = []
    for new_combin in new_combin_list:
        if any(list(map(issuperset, repeat(new_combin), new_itemsets))):
            selected_combin.append(new_combin)
    # 2nd. Check the k combinations with the k-1 infrequent combinations
    # To reduce the no. of loop, use back combinations(k)
    needed_combin = []
    for combin in selected_combin:
        if not any([list(sub_combin) in infrequent
                    for sub_combin in combinations(combin, k-1)]):
            needed_combin.append(combin)
    return needed_combin

def build_hash_tree(hash_combin_list, k):
    # Build hash tree with hashed combin list
    tree = lambda: defaultdict(tree)
    hash_tree = tree()
    for hash_combin in hash_combin_list:
        current_level = hash_tree[hash_combin[0]]
        i = 1
        while not i == k-1:
            current_level = current_level[hash_combin[i]]
            i += 1
        current_level[hash_combin[-1]] = 0
    return hash_tree
            
def back_hash_with_frequent_items(itemset, frequent_items):
    # Back hash the input (hashed frequent itemsets) to ensure 
    # the returns all are correct and in the frequent items
    bh_itemset = []
    for item in itemset:
        bh_item = item
        while not bh_item in frequent_items:
            bh_item = str(int(bh_item) + hash_value)
        bh_itemset.append(bh_item)
    return bh_itemset

def is_frequent_itemset(hash_tree, data_list, frequent_items, \
                        hash_combin_list, min_sup, k):
    # Check the transcations with the items of all combinations
    # from frequent itemsets
    for data_transcation in data_list:
        transcation = data_transcation.copy()
        for item in data_transcation:
            if not item in frequent_items:
                transcation.remove(item)
        # Check the remaining items in transcation
        # And then count the combinations in the hash tree
        if len(transcation) >= k:
            hash_transcation = hash_it(transcation)
            hash_combinations = [*map(list, combinations(hash_transcation, k))]
            if k > 2: # Withdraw unnecessary combinations after 2nd run
                hash_combinations = [hash_combin 
                                     for hash_combin in hash_combinations
                                     if hash_combin in hash_combin_list]
            for hash_combin in hash_combinations:
                current_level = hash_tree[hash_combin[0]]
                i = 1
                while not i == k-1:
                    current_level = current_level[hash_combin[i]]
                    i += 1
                current_level[hash_combin[-1]] += 1
    # Check the value of the bucket which is larget than min_sup or not
    new_itemsets = []
    for hash_combin in hash_combin_list:
        current_level = hash_tree[hash_combin[0]]
        i = 1
        while not i == k-1:
            current_level = current_level[hash_combin[i]]
            i += 1
        if current_level[hash_combin[-1]] >= min_sup:
            itemset = [str(item) for item in hash_combin]
            itemset = back_hash_with_frequent_items(itemset, frequent_items)
            new_itemsets.append(itemset)
    return new_itemsets

def Hash_Tree(file):
    # Find all frequent itemsets with Apriori Algorithm and Hash Tree
    starttime = time()
    data_list = load_data(file)
    new_itemsets = initialize(data_list)
    frequent_itemsets = new_itemsets
    k = 2
    while new_itemsets:
        frequent_items, hash_combin_list, new_combin_list = \
        form_combination(new_itemsets, k)
        if k > 2:
            infrequent = infrequent_combination(old_combin_list, new_itemsets)
            needed_combin_list = \
            prune_combination(new_combin_list, infrequent, new_itemsets, k)
            hash_needed_combin_list = sorted([*map(hash_it, needed_combin_list)])
            hash_tree = build_hash_tree(hash_needed_combin_list, k)
            new_itemsets = is_frequent_itemset(hash_tree, data_list, \
                           frequent_items, hash_needed_combin_list, min_sup, k)
        else:
            hash_tree = build_hash_tree(hash_combin_list, k)
            new_itemsets = is_frequent_itemset(hash_tree, data_list, \
                           frequent_items, hash_combin_list, min_sup, k)
        old_combin_list = new_combin_list
        frequent_itemsets += new_itemsets
        k += 1
    return frequent_itemsets, starttime
    
# Run Hash Tree and time it 
frequent_itemsets, starttime = Hash_Tree('a1dataset.txt')
frequent_itemsets_list = sort_it(frequent_itemsets)
frequent_itemsets_list = sorted(frequent_itemsets_list, key=len)
# Print the result and running time
print('Running Time: {}s\nFrequent itemsets are {}:\n{}'.format(
        (time() - starttime), len(frequent_itemsets_list), 
        frequent_itemsets_list))

            
            
    
        
            