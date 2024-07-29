#modify npy feature vectors to include selected features

import numpy as np
import os
with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    
pdb_list = []

for data in data_list:
    pdb_list.append(data.strip().split('\t')[1].lower())

def normalize(x):
    max = np.max(x)
    min = np.min(x)
    if (abs(max-min) < 1e-7):
        return x-min #reduce overflow, possibility of all zeros
    else:
        return (x-min)/(max-min)
def normalize_pdbs():
    count = 0
    for pdb in pdb_list: #4034 or something
    #pdb id = pdb
        if (count % 100 == 0):
            print("normalized " + str(count) + " of " + str(len(pdb_list)) + " PDBs")
        #best practice to not overwrite data
        #if(os.path.exists("codebase/data/npy/" + str(pdb) + "-node_feat_normalized.npy")):
            #already processed
        #    print("already processed file " + str(count) + "(" + str(pdb) + ")")
        #    continue
        try:
            node_feat = np.load("codebase/data/npy/" + str(pdb) + "-node_feat.npy") #should be (n, 2256) where n is # of available residues in PDB
            node_feat_norm = np.zeros((node_feat.shape[0], 1024+1218+14)) #PLM + DSSP
            #print(str(np.shape(node_feat[:,:1024])) + " " + str(np.shape(node_feat[:,2242:])))
            for i in range(len(node_feat)):
                prott5 = node_feat[i,:1024]
                ac = node_feat[i, 1024:1234]
                pseaac = node_feat[i, 1234:1269]
                ct = node_feat[i, 1269:1612]
                ld = node_feat[i, 1612:2242]
                dssp = node_feat[i, 2242:] #don't normalize, already between 0 and 1
                #print(str(np.shape(prott5)) + " " + str(np.shape(ac)) + " " + str(np.shape(pseaac)) + " " + str(np.shape(ct)) + " " + str(np.shape(ld)) + " " + str(np.shape(dssp)))
                #should be 1024 210 35 343 630
                #print(str(np.max(dssp)))
                prott5 = normalize(prott5).copy()
                ac = normalize(ac).copy()
                pseaac = normalize(pseaac).copy()
                ct = normalize(ct).copy()
                ld = normalize(ld).copy()
                node_feat_norm[i,:] = np.concatenate((prott5, ac, pseaac, ct, ld, dssp))
            #print(str(np.shape(node_feat_1dmf_remove)))
            count += 1
            if(np.max(node_feat_norm) > 1.1):
                print("normalization failed for " + str(pdb) + ", max is " + str(np.max(node_feat_norm)))
                break
            np.save("codebase/data/npy/" + str(pdb) + "-node_feat_normalized.npy", node_feat_norm)
            #break #for testing purposes, then run on everything w/ continue
        except Exception as e:
            #no file available
            print(str(e))
            print("no data available (file generation failed) for pdb " + str(pdb))
            continue

def reduce_dim():
    count = 0
    for pdb in pdb_list: 
        if (count % 1000 == 0):
            print("reprocessed " + str(count) + " of " + str(len(pdb_list)) + " PDBs (normalized)")
        count += 1
        #if(os.path.exists("codebase/data/npy/" + str(pdb) + "-node_feat_reduced.npy")):
            #already processed
            #os.remove("codebase/data/npy/" + str(pdb) + "-node_feat_plm_dssp.npy")
            #print("already processed file " + str(count) + "(" + str(pdb) + ")")
            #continue
            #node_feat = np.load("codebase/data/npy/" + str(pdb) + "-node_feat_reduced.npy")
            #print(np.shape(node_feat))
            #break
        node_feat = np.load("codebase/data/npy/" + str(pdb) + "-node_feat_normalized.npy") #should be (n, 2256) where n is # of available residues in PDB
        node_feat_reduced = np.zeros((node_feat.shape[0], 1024+14)) #1024 for PLM, 1024+14 for PLM+DSSP, 1024+1218 for PLM+1DMF, 1024+1218+14 for PLM+1DMF+DSSP
        #print(str(np.shape(node_feat[:,:1024])) + " " + str(np.shape(node_feat[:,2242:])))
        node_feat_reduced[:,:1024] = node_feat[:,:1024] #PLM, should save as node_feat_reduced
        node_feat_reduced[:,1024:] = node_feat[:,2242:] #PLM+DSSP (along with previous line), should save as node_feat_reduced_dssp
        #PLM+1DMF+DSSP: use node_feat_normalized

        np.save("codebase/data/npy/" + str(pdb) + "-node_feat_reduced_dssp.npy", node_feat_reduced)
        