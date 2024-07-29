
import requests
import time
import os
import modules
import numpy as np
import torch

#calculate the average connectivity of extracted hybrid-feature protein graph representations

with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    
pdb_list = []
for data in data_list:
    pdb_list.append(data.strip().split('\t')[1].lower())

small_protein = 0
avg_nodes = 0.0
edge_oob = 0
proteins = 0.0

avg_edges_per_node_6 = 0
avg_edges_per_node_7 = 0
avg_edges_per_node_8 = 0
avg_edges_per_node_9 = 0
avg_edges_per_node_10 = 0

avg_edges_per_node_6_5 = 0
avg_edges_per_node_7_5 = 0
avg_edges_per_node_8_5 = 0
avg_edges_per_node_9_5 = 0

for pdb in pdb_list: 
    node_feat = np.load("codebase/data/npy/" + str(pdb) + "-node_feat_reduced_dssp.npy") #should be (n, 1024+14) where n is # of available residues in PDB
    edge_list_6 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_6.npy") #should be (2, E) where E is # edges
    edge_list_7 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_7.npy") #should be (2, E) where E is # edges
    edge_list_8 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_8.npy") #should be (2, E) where E is # edges
    edge_list_9 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_9.npy") #should be (2, E) where E is # edges
    edge_list_10 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_10.npy") #should be (2, E) where E is # edges
    edge_list_6_5 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_6_5.npy") #should be (2, E) where E is # edges
    edge_list_7_5 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_7_5.npy") #should be (2, E) where E is # edges
    edge_list_8_5 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_8_5.npy") #should be (2, E) where E is # edges
    edge_list_9_5 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_9_5.npy") #should be (2, E) where E is # edges

    node_feat_shape = np.shape(node_feat)
    edge_list_6_shape = np.shape(edge_list_6)
    edge_list_7_shape = np.shape(edge_list_7)
    edge_list_8_shape = np.shape(edge_list_8)
    edge_list_9_shape = np.shape(edge_list_9)
    edge_list_10_shape = np.shape(edge_list_10)
    edge_list_6_5_shape = np.shape(edge_list_6_5)
    edge_list_7_5_shape = np.shape(edge_list_7_5)
    edge_list_8_5_shape = np.shape(edge_list_8_5)
    edge_list_9_5_shape = np.shape(edge_list_9_5)
    
    avg_nodes += node_feat_shape[0]
    avg_edges_per_node_6 += (edge_list_6_shape[1]/node_feat_shape[0])
    avg_edges_per_node_7 += (edge_list_7_shape[1]/node_feat_shape[0])
    avg_edges_per_node_8 += (edge_list_8_shape[1]/node_feat_shape[0])
    avg_edges_per_node_9 += (edge_list_9_shape[1]/node_feat_shape[0])
    avg_edges_per_node_10 += (edge_list_10_shape[1]/node_feat_shape[0])
    avg_edges_per_node_6_5 += (edge_list_6_5_shape[1]/node_feat_shape[0])
    avg_edges_per_node_7_5 += (edge_list_7_5_shape[1]/node_feat_shape[0])
    avg_edges_per_node_8_5 += (edge_list_8_5_shape[1]/node_feat_shape[0])
    avg_edges_per_node_9_5 += (edge_list_9_5_shape[1]/node_feat_shape[0])
    
    proteins += 1
    if (node_feat_shape[0] <= 10):
        small_protein += 1  

avg_edges_per_node_6 = (avg_edges_per_node_6 / proteins)
avg_edges_per_node_7 = (avg_edges_per_node_7 / proteins)
avg_edges_per_node_8 = (avg_edges_per_node_8 / proteins)
avg_edges_per_node_9 = (avg_edges_per_node_9 / proteins)
avg_edges_per_node_10 = (avg_edges_per_node_10 / proteins)
avg_edges_per_node_6_5 = (avg_edges_per_node_6_5 / proteins)
avg_edges_per_node_7_5 = (avg_edges_per_node_7_5 / proteins)
avg_edges_per_node_8_5 = (avg_edges_per_node_8_5 / proteins)
avg_edges_per_node_9_5 = (avg_edges_per_node_9_5 / proteins)
avg_nodes = (avg_nodes / proteins)

print("average edges per node (6): " + str(avg_edges_per_node_6))
print("average edges per node (6.5): " + str(avg_edges_per_node_6_5))
print("average edges per node (7): " + str(avg_edges_per_node_7))
print("average edges per node (7.5): " + str(avg_edges_per_node_7_5))
print("average edges per node (8): " + str(avg_edges_per_node_8))
print("average edges per node (8.5): " + str(avg_edges_per_node_8_5))
print("average edges per node (9): " + str(avg_edges_per_node_9))
print("average edges per node (9.5): " + str(avg_edges_per_node_9_5))
print("average edges per node (10): " + str(avg_edges_per_node_10))
print("average amino acids per protein: " + str(avg_nodes))
print("total proteins: " + str(proteins))
print("small proteins: " + str(small_protein))