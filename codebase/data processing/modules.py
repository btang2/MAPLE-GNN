import os
import sys
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio import SeqIO
from Bio.PDB import PDBList
import numpy as np
import numpy.linalg as la
#import scipy as sp
import networkx as nx
import requests as r
import matplotlib.pyplot as plt
import warnings
from transformers import T5EncoderModel, T5Tokenizer
import torch
import re
import time
from helpers import feat_engg_manual_main
import matplotlib.patches as mpatches 
import py3Dmol

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#from pathlib import Path
#path_root = Path(__file__).parents[1]  # upto 'codebase' folder
#sys.path.insert(0, str(path_root))
# print(sys.path)

def PDB_to_DSSP(ID, seq_dict):
    seq_key_list = list(seq_dict.keys())
    #convert PDB to DSSP file (modernized code from Wei et. al. (2023) DeepProSite) with integration from Biopython PDB.DSSP functionality
    #since we are directly using sequence data from REST API, no need to align structures
    #aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    #rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170, 185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    #Default DSSP output (unprocessed): # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
    # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
    # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)

    #QUIET = True -- suppress warnings
    p = PDBParser(QUIET=True)
    structure = p.get_structure(id=ID, file="codebase/data/pdb/" + str(ID) + ".pdb") #get PDB
    model = structure[0]
    dssp = DSSP(model=model, in_file="codebase/data/pdb/" + str(ID) + ".pdb", dssp="mkdssp") #use Biopython & mkdsssp to create DSSP dictionary

    dssp_dict = {}
    #print(str(dssp.keys()))
    default_dssp = torch.tensor([[0.0,     0.0,     1.0,     1.0,   0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).to(device) #for when AA from sequence cannot be found in PDB, may not be needed now that we are extracting directly from PDB
    #                       sin(phi) sin(psi) cos(phi) cos(psi) rASA  1    2    3    4    5    6    7    8    9
    for chain_id in seq_key_list:
        dssp_feature = default_dssp.repeat(len(seq_dict[chain_id]), 1) #create DSSP feature matrix (n x 14) & save to dict
        #print(np.shape(dssp_feature))
        dssp_dict[chain_id] = dssp_feature
    
    #cnt = 0
    #chain_id = list(dssp.keys())[0][0] #should default to primary chain if working through multiple chains
    chain_ind_cnt = {}
    for chain_id in seq_key_list:
        chain_ind_cnt[chain_id] = 0
    for a_key in list(dssp.keys()):
        chain_id = a_key[0] #this should work as a handle to access dssp_dict
        #print(chain_id)
        try:
            #aa_idx = dssp[a_key][0] - 1 #index of amino acid (-1 to convert from 1-indexing to 0-indexing), from single-chain
            aa_idx = chain_ind_cnt[chain_id] #index of amino acid with unknowns removed
            #print(aa_idx)
            aa = dssp[a_key][1] #retrieve aminoacid ID, if X, assign default DSSP
            if (aa == 'X'):
                continue
            SS = dssp[a_key][2]
            if SS == "-":
                SS = "C" #none secondary structure, but not unknown
            SS_vec = np.zeros(9) #last dimension represents "unknown" for missing residue (default)
            SS_vec[SS_type.find(SS)] = 1.0
            PHI = float(dssp[a_key][4]) * np.pi / 180 #default value 2pi
            PSI = float(dssp[a_key][5]) * np.pi / 180 #default value 2pi
            rASA = float(dssp[a_key][3]) #default value 0
            aa_feat = torch.cat((torch.tensor([np.sin(PHI), np.sin(PSI), np.cos(PHI), np.cos(PSI), rASA]).to(device), torch.tensor(SS_vec).to(device))) #concat to form DSSP for specific node (CUDA-accelerated) 
            dssp_dict[chain_id][aa_idx] = aa_feat #update DSSP feature matrix
            chain_ind_cnt[chain_id] += 1
            #cnt += 1
            
        except Exception as e:
            print("!! DSSP Exception: " + str(e))
            continue
            #print("exception reached for chain " + chain_id)
            #continue
    for chain_id in seq_key_list:
        print("Chain " + chain_id + ": DSSP successfully calculated for " + str(chain_ind_cnt[chain_id]) + " residues")
    return dssp_dict
    
def seq_to_PLM_embedding(seq_dict):
    seq_key_list = list(seq_dict.keys())
    #adapted from ProtTrans5 Quickstart Guide
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    if device==torch.device("cpu"):
        model.to(torch.float32) 

    #protein sequence preprepared
    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = []
    for i in seq_key_list:
        sequence_examples.append(seq_dict[i])
    #print(sequence_examples)
    #print(len(sequence_examples[0]))
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:len(prot_seq)]) 
    emb_dict = {}
    for i in range(len(list(seq_dict.keys()))):
        emb_i = embedding_repr.last_hidden_state[i,:len(seq_dict[seq_key_list[i]])] # shape (len(prot_seq) x 1024)
        emb_dict[seq_key_list[i]] = emb_i
    return emb_dict

    # if you want to derive a single representation (per-protein embedding) for the whole protein
    #emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)

def ID_to_essentials(ID):
    seq_dict = {}
    for record in SeqIO.parse("codebase/data/pdb/" + str(ID) + ".pdb", "pdb-atom"):
        seq_dict[record.annotations["chain"]] = str(record.seq)
    seq_key_list = list(seq_dict.keys())
    seq_lenptr = np.zeros(len(seq_key_list) + 1, dtype=int) #similar to CSR rowptr; shows where each sequence starts and stops
    seq_lenptr[0] = 0
    seq_key_to_idx = {}
    for i in range(len(seq_key_list)):
        chain_id = seq_key_list[i]
        seq_key_to_idx[chain_id] = i
        seq_lenptr[i+1] = seq_lenptr[i] + len(seq_dict[chain_id]) #specifically, chain_id of seq_key_list[i] goes from index seq_lenptr[i] to seq_lenptr[i+1]-1
        #& notably, seq_lenptr[-1] = length of the whole concatenated sequence
    print("seq_lenptr: " + str(seq_lenptr))
    return seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx

def PDB_to_coord_dict(ID, seq_dict, seq_key_to_idx, seq_lenptr):
    #first, convert PDB to list of coords
    #how to "average" coords of atoms to obtain coord of residue? use alpha-carbon
    p = PDBParser(QUIET=True)
    structure = p.get_structure(id=ID, file="codebase/data/pdb/" + str(ID) + ".pdb") #get PDB
    model = structure[0]
    coord_dict = {}
    cur_idx = 0
    print("Extracting Chain Information -- START")
    for chain in model:
        #print("[" + str(seq_key_to_idx[chain.id]) + "] CHAIN " + str(chain.id) + " & SEQ LEN " + str(len(seq_dict[chain.id])))
        for residue in list(chain):
            hetero_flag = residue.id[0].strip()
            # Empty strings evaluate to False. hetero_flag returns False if the residue is not a water molecule, and should also get rid of heteroatoms
            if not hetero_flag:
                #chain.detach_child(residue.id) 
                #print(str(residue.id[1]) + " " + str(residue.resname))
                #print(str(residue['CA'].coord))
                #index would be seq_lenptr[seq_key_to_idx[chain.id]] + residue.id[1] - 1 & coord is residue['CA'].coord but maybe random missing residues, so keep cur_idx updated instead for continuity
                #coord_dict[seq_lenptr[seq_key_to_idx[chain.id]] + residue.id[1] - 1] = residue['CA'].coord
                if(len(residue.resname) != 3):
                    break #not a protein chain
                try:
                    coord_dict[cur_idx] = residue['CA'].coord
                except:
                    #no Carbon Atom???
                    print("NO ALPHA-CARBON at index " + str(cur_idx) + ". Residue name: " + str(residue.resname))
                cur_idx += 1
                #slightly different & somewhat more strict compared to biographs implementation, it is possible that higher cutoff would make the model more accurately represent the protein    
        #if not list(chain):
            #model.detach_child(chain.id)
            #print("PDB file error: chain not list -- skipped")
    print("Extracting Chain Information -- END")
    return coord_dict
            
def generate_node_features(ID, seq_dict, seq_key_list, seq_lenptr):
    #generate ProtTrans T5 LLM embedding
    print("Generating Node Features -- PLM embedding -- START")
    emb_dict = seq_to_PLM_embedding(seq_dict)
    print("Generating Node Features -- PLM Embedding -- END")
    #for chain_id in seq_key_list:
    #    print("chain " + chain_id + " embedding dims: " + str(emb_dict[chain_id].shape))

    #generate 1DMF from sequence
    #possible bug/confounder: AC30 generates NaN values, most likely bug with original fasta file as I cannot seem to reproduce it
    print("Generating Node Features -- 1DMF -- START")
    mf_dict = {}
    feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD']
    for chain_id in seq_key_list:
        seq_manual_feat_dict = feat_engg_manual_main.extract_prot_seq_1D_manual_feat(root_path="helpers/", prot_seq = seq_dict[chain_id], feature_type_lst = feature_type_lst, deviceType='cuda:0')
        mf_feat = torch.cat((torch.tensor(seq_manual_feat_dict['AC30']), torch.tensor(seq_manual_feat_dict['PSAAC15']), torch.tensor(seq_manual_feat_dict['ConjointTriad']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_C']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_T']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'])))
        mf_feat = mf_feat.repeat(len(seq_dict[chain_id]),1).to(device)
        mf_dict[chain_id] = mf_feat
    print("Generating Node Features -- 1DMF -- END")
    #for chain_id in seq_key_list:
    #    print("chain " + chain_id + " 1DMF dims: " + str(mf_dict[chain_id].shape))
    
    print("Generating Node Features -- DSSP -- START")
    dssp_dict = PDB_to_DSSP(ID=ID,seq_dict=seq_dict)
    print("Generating Node Features -- DSSP -- END")
    #print(dssp_dict)
    #print(dssp_feat.shape)
    #construct concatenated node feature matrix (does not depend on)
    print("Generating Node Features -- Feature Vector Concatenation -- START")
    node_feat_matrix = torch.tensor(np.zeros((seq_lenptr[-1], 2256))).to(device)
    for i in range(len(seq_key_list)):
        chain_id = seq_key_list[i]
        #node_feat_dict[chain_id] = torch.cat((emb_dict[chain_id], mf_dict[chain_id], dssp_dict[chain_id]), dim=1).to(device)
        node_feat_matrix[seq_lenptr[i]:seq_lenptr[i+1],] = torch.cat((emb_dict[chain_id], mf_dict[chain_id], dssp_dict[chain_id]), dim=1).to(device)
        #print(str(chain_id) + " feature matrix shape: " + str(node_feat_dict[chain_id].shape))
        #node_feat_matrix = torch.cat((node_feat_matrix, node_feat_dict[chain_id]),dim=0)
    print("Generating Node Features -- Feature Vector Concatenation -- END")
    return node_feat_matrix

    #generate Node Feature matrix (n x 2256), concat wrt seq_key_list order; since all features generated per-chain the order should not affect the graph implementation
    
    #node_features = torch.cat((llm_feat,mf_feat, dssp_feat), dim=1)
    #print ("Node Feature Matrix Dimensions:  " + str(node_features.shape))

def generate_edge_features(coord_dict, angstrom_cutoff):
    #this purely depends on the coordinates; everything else is encoded in indices/etc already, runtime should be o(n^2)
    #adj_matrix = torch.tensor(np.zeros((len(coord_dict), len(coord_dict)), dtype=int)).to(device)
    #edge_features = torch.tensor(np.zeros((len(coord_dict), len(coord_dict), 2))).to(device) #features are spatial distance encoding & sequence distance encoding
    edge_list = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features = torch.empty(0,2).to(device)
    for i in range(len(coord_dict)):
        for j in range(i, len(coord_dict)):
            if (i == j):
                dist_ij = la.norm(coord_dict[i] - coord_dict[j], ord=2)
                edge_list = torch.cat((edge_list, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features = torch.cat((edge_features, torch.from_numpy(np.array([[(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                #adj_matrix[i][j] = 1 #i=j, just process once
                #edge_features[i][j] = torch.tensor([(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]).to(device)
            else:
                dist_ij = la.norm(coord_dict[i] - coord_dict[j], ord=2)
                if (dist_ij < angstrom_cutoff):
                    #edge
                    edge_list = torch.cat((edge_list, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features = torch.cat((edge_features, torch.from_numpy(np.array([[(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list = torch.cat((edge_list, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features = torch.cat((edge_features, torch.from_numpy(np.array([[(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    #adj_matrix[i][j] = 1
                    #adj_matrix[j][i] = 1
                    #edge_features[i][j] = torch.tensor([(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]).to(device)
                    #edge_features[i][j] = torch.tensor([(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]).to(device)
                else:
                    #no edge
                    no_edge = True
                    #edge_features[i][j] = torch.tensor([0,0]).to(device)
                    #edge_features[j][i] = torch.tensor([0,0]).to(device)
    return edge_list.T, edge_features

def save_features(ID, seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx, ang_cutoff):
    #save node and edge features in .npy compressed format to avoid recomputation
    print("Saving Features for Protein: " + str(ID))
    print("Generating Node Features -- START")
    node_feat_matrix = generate_node_features(ID=ID, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_lenptr=seq_lenptr) #that's so clean
    print("Generating Node Features -- END")
    coord_dict = PDB_to_coord_dict(ID=ID, seq_dict=seq_dict, seq_key_to_idx = seq_key_to_idx, seq_lenptr=seq_lenptr)
    print("Generating Adjacency Matrix and Edge Features -- START")
    edge_list, edge_features = generate_edge_features(coord_dict=coord_dict, angstrom_cutoff=ang_cutoff) #angstrom cutoff should be float
    print("Generating Adjacency Matrix and Edge Features -- END")
    print("Saving Results to numpy -- START")
    np.save("codebase/data/npy/" + str(ID) + "-node_feat.npy", node_feat_matrix.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list.npy", edge_list.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat.npy", edge_features.cpu().numpy()) 
    print("Saving Results to numpy -- END")
    #store adj_matrix in house to avoid recomputing; eventually it'll all be from here
    

def visualize_graph(ID, ang_cutoff, adj_matrix, seq_lenptr, seq_key_list):
    #note input is ADJACENCY MATRIX, not EDGE LIST
    plt.figure(figsize=(18,18))
    G = nx.from_numpy_array(adj_matrix) #convert to CPU for graph visualization
    G.remove_edges_from(nx.selfloop_edges(G)) #remove self-loops for better visualizations (self-loops implied)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    color_map = []
    pos = nx.spring_layout(G)
    cur_comp = 0
    for node in G:
        if node > seq_lenptr[cur_comp+1]:
            cur_comp += 1
        color_map.append(colors[cur_comp])
    
    plt.title(str(ID) + " Protein Graph Visualization (cutoff=" + str(ang_cutoff) + ")", size=30)
    patch_arr = []
    for i in range(len(seq_key_list)):
        patch_arr.append(mpatches.Patch(color=colors[i], label='Chain ' + str(seq_key_list[i])))
    plt.legend(handles=patch_arr,loc=1, prop={'size':18})

    n_nodes = G.number_of_nodes()
    #draw graph based on overall size: small (<=250), medium (>250, <= 750), large (>750)
    #average human protein is <500 so would fall into small or medium category
    width = 1
    node_size = 500
    font_size = 50
    if (n_nodes <= 250):
        #small
        width = 1.0
        node_size = 1200
        font_size = 24
    elif (n_nodes > 250 and n_nodes <= 750):
        width = 0.8
        node_size = 300
        font_size = 8
    else:
        #n_nodes > 750
        width = 0.7
        node_size = 100
        font_size = 5
    nx.draw_networkx_edges(G=G, pos=pos,width=width)
    nx.draw_networkx_nodes(G=G, pos=pos, node_color = color_map, node_size=node_size)
    nx.draw_networkx_labels(G=G, pos=pos, font_size = font_size)

    strFile = "codebase/images/" + str(ID) + ".png"
    if os.path.isfile(strFile):
        os.remove(strFile)   # Opt.: os.system("rm "+strFile)
    
    plt.box(False) #remove margin box
    plt.subplots_adjust(top=0.95) #better visuals
    plt.savefig(strFile, bbox_inches = "tight")

if __name__ == '__main__':
    #get_dssp(ID="1tup", ref_seq="SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNT")
    #print(PDB_to_DSSP("1tup", "codebase/data/pdb/1tup.pdb"))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Protein PDB id & sequence from ID
    ID = "3i40" #insulin (2-chain, small)
    #ID = "1bfg" #fibroblast growth factor 1 (1-chain, small)
    #ID = "1a3n" #hemoglobin (4-chain, medium)
    #ID = "1tup" #P53 tumor suppressor (3-chain, medium)
    #ID = "3os8" #estrogen receptor (4-chain, large)
    seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx = ID_to_essentials(ID=str(ID))
    ang_cutoff = 6.0 #constant for now, seems reasonable for academia & visualization
    
    #nodefeat = np.load("codebase/data/npy/" + ID + "-node_feat.npy")
    #for i in range(len(nodefeat[0])):
    #    if (np.isnan(nodefeat[0][i])):
    #        print(str(i)) 
    #print(np.count_nonzero(np.isnan(nodefeat))) 1680

    #only need to run once per protein (although must keep ang_cutoff the same)
    save_features(ID=ID, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_lenptr=seq_lenptr, seq_key_to_idx=seq_key_to_idx, ang_cutoff=ang_cutoff) #codebase/data/npy/ID-node_feat or ID-edge_feat or ID-adj
    edge_list = np.load("codebase/data/npy/" + str(ID) + "-edge_list.npy")
    print(str(np.shape(edge_list)))
    #print("Visualizing Graph")
    #adj_matrix = np.load("codebase/data/npy/" + str(ID) + "-adj.npy")
    #visualize_graph(ID=ID, ang_cutoff=ang_cutoff, adj_matrix= adj_matrix, seq_lenptr=seq_lenptr, seq_key_list=seq_key_list)
    



    
    
    


