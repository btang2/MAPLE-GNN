#modify extracted explanaiton npy for correct feature input
import numpy as np
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
explanation_pairs = [] #insert explanation pairs here
#[("1TUP", "A"), ("1TUP", "B"), ("6EC0", "A"), ("6EC0", "B"), ("1GRI", "A"), ("1GRI", "B"), ("1BUW", "C"), ("1BUW", "D"), ("3OS8", "A"), ("3OS8", "C"), ("1RFB", "A"), ("1RFB", "B"), ("1AGN", "C"), ("1AGN", "D"), ("5I6Z", "B"), ("5I6Z", "C"), ("7AOS", "A"), ("7AOS", "B"), ("1BJ1", "H"), ("1BJ1", "L"), ("8J7F", "C"), ("8J7F", "D"), ("3I40", "A"), ("3I40", "B"), ("6DDF", "A"), ("6DDF", "R"), ("1IZN", "C"), ("1IZN", "D"), ("1FIN", "A"), ("1FIN", "B"), ("1RUZ", "L"), ("1RUZ", "M"), ("7K43", "B"), ("7K43", "C"), ("2AAI", "A"), ("2AAI", "B"), ("3IFL", "H"), ("3IFL", "L"), ("1BR1", "A"), ("1BR1", "B"), ("3RU8", "H"), ("3RU8", "X")]

def normalize(x):
    max = np.max(x)
    min = np.min(x)
    if (abs(max-min) < 1e-7):
        return x-min #reduce overflow, possibility of all zeros
    else:
        return (x-min)/(max-min)
def normalize_pdbs():
    count = 0
    for (ID, chainid) in explanation_pairs: 
        pdb = str(ID.lower()) + "-" + str(chainid)
        if (count % 100 == 0):
            print("normalized " + str(count) + " of " + str(len(explanation_pairs)) + " PDBs")
        #best practice to not overwrite data
        #if(os.path.exists("codebase/data/npy/" + str(pdb) + "-node_feat_normalized.npy")):
            #already processed
        #    print("already processed file " + str(count) + "(" + str(pdb) + ")")
        #    continue
        try:
            node_feat = np.load("codebase/data/explain-npy/" + str(pdb) + "-node_feat.npy") #should be (n, 2256) where n is # of available residues in PDB
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
            np.save("codebase/data/npy/explain-npy" + str(pdb) + "-node_feat_normalized.npy", node_feat_norm)
            #break #for testing purposes, then run on everything w/ continue
        except Exception as e:
            #no file available
            print(str(e))
            print("no data available (file generation failed) for pdb " + str(pdb))
            continue

def reduce_dim():
    
    for (ID, chainid) in explanation_pairs: 
        pdb = str(ID.lower()) + "-" + str(chainid)
        node_feat = np.load("codebase/data/explain-npy/" + str(pdb) + "-node_feat_normalized.npy") #should be (n, 2256) where n is # of available residues in PDB
        node_feat_1dmf_remove = np.zeros((node_feat.shape[0], 1024+14)) #1024 for PLM, 1024+14 for PLM+DSSP, 1024+1218 for PLM+1DMF, 1024+1218+14 for PLM+1DMF+DSSP
        #print(str(np.shape(node_feat[:,:1024])) + " " + str(np.shape(node_feat[:,2242:])))
        node_feat_1dmf_remove[:,:1024] = node_feat[:,:1024] #PLM
        node_feat_1dmf_remove[:,1024:] = node_feat[:,2242:] #PLM+DSSP (along with previous line)
        #node_feat_1dmf_remove[:,:1024+1218] = node_feat[:,:1024+1218] #PLM+1DMF
        #PLM+1DMF+DSSP: use node_feat_normalized
        
        #print(str(np.shape(node_feat_1dmf_remove)))
        np.save("codebase/data/explain-npy/" + str(pdb) + "-node_feat_reduced_dssp.npy", node_feat_1dmf_remove)
