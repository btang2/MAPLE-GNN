import requests
import time
import os
import modules
import math
import torch
from zipfile import ZipFile

#zip relevant extracted data for future use

with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    
pdb_set = set()

for data in data_list:
    pdb_set.add(data.strip().split('\t')[1].lower())



#zf = ZipFile('graphrepresentations.zip', 'w')
#for pdb in pdb_set:
#    zf.write(f"codebase/data/npy/{pdb}-node_feat_reduced_dssp.npy")
#    zf.write(f"codebase/data/npy/{pdb}-edge_list_9.npy")
#    zf.write(f"codebase/data/npy/{pdb}-edge_feat_9.npy")

zf = ZipFile('pdb-files.zip', 'w')
for pdb in pdb_set:
    zf.write(f"codebase/data/pdb/{pdb}.pdb")

zf.close()