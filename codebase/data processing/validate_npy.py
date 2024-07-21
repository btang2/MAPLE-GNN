
import requests
import time
import os
import modules
import numpy as np
import torch


with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    
pdb_list = []

def perc_err(num_err):
    #calculate percentage of errors
    return np.round((100.0 * num_err) / 4034.0, 2)

for data in data_list:
    pdb_list.append(data.strip().split('\t')[1].lower())


node_feat_err = 0
edge_list_err = 0
edge_feat_err = 0
edge_agree_err = 0
more_edge_feat = 0
low_edges = 0
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

#edge_oob_list = []
for pdb in pdb_list: #4034 or something
    #pdb id = pdb
    #node_feat = np.load("codebase/data/npy/" + str(pdb) + "-node_feat.npy") #should be (n, 2256) where n is # of available residues in PDB
    node_feat = np.load("codebase/data/npy/" + str(pdb) + "-node_feat_normalized.npy") #should be (n, 2256) where n is # of available residues in PDB
    edge_list_6 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_6.npy") #should be (2, E) where E is # edges
    edge_list_7 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_7.npy") #should be (2, E) where E is # edges
    edge_list_8 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_8.npy") #should be (2, E) where E is # edges
    edge_list_9 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_9.npy") #should be (2, E) where E is # edges
    edge_list_10 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_10.npy") #should be (2, E) where E is # edges
    edge_list_6_5 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_6_5.npy") #should be (2, E) where E is # edges
    edge_list_7_5 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_7_5.npy") #should be (2, E) where E is # edges
    edge_list_8_5 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_8_5.npy") #should be (2, E) where E is # edges
    edge_list_9_5 = np.load("codebase/data/npy/" + str(pdb) + "-edge_list_9_5.npy") #should be (2, E) where E is # edges

    #edge_feat_6 = np.load("codebase/data/npy/" + str(pdb) + "-edge_feat.npy") #should be (E, 2) where e is # edges, 2 is len of feature vec
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
    #if (edge_list.max() > node_feat_shape[0]):
        #print("edge list out of bounds... max index is " + str(edge_list.max()) + " while max node is " + str(node_feat_shape[0]))
        #edge_oob += 1
        #edge_oob_list.append(pdb)
    #if (edge_list_shape[1] <= 1000):
        #print("LowEdges: PDB " + str(pdb) + " has " + str(edge_list_shape[1]) + " edges for " + str(node_feat_shape[0]) + " nodes")
        #low_edges += 1
    if (node_feat_shape[1] != 2256 or node_feat_shape[0] == 0):
        #print("PROCESSING ERROR -- pdb " + str(pdb) + " has irregular node_feat with shape " + str(node_feat_shape) + ", should be (n,2256)") 
        node_feat_err += 1
    #if (edge_list_shape[0] != 2):
        #print("PROCESSING ERROR -- pdb " + str(pdb) + " has irregular edge_list with shape " + str(edge_list_shape) + ", should be (2, E)")
        #edge_list_err += 1
    #if (edge_feat_shape[1] != 2):
        #print("PROCESSING ERROR -- pdb " + str(pdb) + " has irregular edge_feat with shape " + str(edge_feat_shape) + ", should be (E, 2)")
        #edge_feat_err += 1
    #if (edge_list_shape[1] != edge_feat_shape[0]):
        #print("PROCESSING ERROR -- pdb " + str(pdb) + " does not have matching number of edges (edge_list_shape[1] and edge_feat_shape[0] do not match), edge_list has " + str(edge_list_shape[1]) + " edges, while edge_feat has " + str(edge_feat_shape[0]) + " edges")
        #edge_agree_err += 1
        #if (edge_feat_shape[0] > edge_list_shape[1]):
        #    more_edge_feat += 1
        

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

#print("out of 4034 PDBs, " + str(node_feat_err) +  " (" + str(perc_err(node_feat_err)) + "%) had node feature errors, " + str(edge_list_err) + " (" + str(perc_err(edge_list_err)) + "%) had edge list errors, " + str(edge_feat_err) + " (" + str(perc_err(edge_feat_err)) + ") had edge feature errors, and " + str(edge_agree_err) + " (" + str(perc_err(edge_agree_err)) + "%) had edge number agreement errors.")
#print("out of" + str(edge_agree_err) + " edge misaligment errors, " + str(more_edge_feat) + " had more edge features than edges and " + str(edge_agree_err - more_edge_feat) + " had less edge features than edges.")
#print("out of 4034 PDBs, " + str(edge_oob) + " (" + str(perc_err(edge_oob)) + "%) had edge misaligment errors.")
#print("out of 4034 PDBs, " + str(no_file) + " (" + str(perc_err(no_file)) + ") had no successful saved data available.") 
#print("out of 4034 PDBs, " + str(low_edges) + " had low (<1000) edges")
#print("average edges per node (amino acid residue): " + str(avg_edges_per_node_per_protein))
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
print("small proteins: " + str(small_protein))
print("node error: " + str(node_feat_err))
#print("edge OOB list: " + str(edge_oob_list))
#hopefully fixed, but will have to check