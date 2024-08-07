import os
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import requests as r
import re
import numpy as np
import numpy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt
from transformers import T5EncoderModel, T5Tokenizer
import torch
from helpers import feat_engg_manual_main
import matplotlib.patches as mpatches 

#main module for data processing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ID_to_essentials(ID, chainid, explain = False):
    #parse PDB and extract essential information
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    seq_dict = {}
    p = PDBParser(QUIET=True)
    chain_found = False
    if (explain):
        structure = p.get_structure(id=ID, file = "codebase/data/explain-pdb/" + str(ID) + ".pdb")
    else:
        structure = p.get_structure(id=ID, file = "codebase/data/pdb/" + str(ID) + ".pdb") #get PDB
    model = structure[0]
    residue_to_remove = []
    chain_to_remove = []
    for chain in model:
        if (chain.id == chainid):
            chain_found = True
        seq = ""
        for residue in chain:
            hetero_flag = residue.id[0].strip()
            if not hetero_flag:
                if (residue.resname in d3to1.keys()):
                    seq += d3to1[residue.resname]
                elif(len(residue.resname) != 3):
                    break 
                else:
                    seq += 'X'
        if(len(seq) <= 1):
            chain_to_remove.append(chain.id) #chain of hetatm's/extraneous chain 
        else:
            seq_dict[chain.id] = str(seq)

    for residue in list(residue_to_remove):
        model[residue[0]].detach_child(residue[1])

    for chain in list(chain_to_remove):
        model.detach_child(chain)

    seq_key_list = list(seq_dict.keys())
    seq_lenptr = np.zeros(len(seq_key_list) + 1, dtype=int) #similar to CSR rowptr; shows where each sequence starts and stops
    seq_lenptr[0] = 0
    seq_key_to_idx = {}
    for i in range(len(seq_key_list)):
        chain_id = seq_key_list[i]
        seq_key_to_idx[chain_id] = i
        seq_lenptr[i+1] = seq_lenptr[i] + len(seq_dict[chain_id]) #specifically, chain_id of seq_key_list[i] goes from index seq_lenptr[i] to seq_lenptr[i+1]-1
        #& notably, seq_lenptr[-1] = length of the whole concatenated sequence
    #print("seq_lenptr: " + str(seq_lenptr))
    #print(seq_dict)
    #print(seq_lenptr)
    if (not chain_found):
        raise Exception("Chain not found: PDB ID " + str(ID) + ", attempted chain: " + str(chainid))
    
    return seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx

def PDB_to_DSSP(ID, chainid, seq_dict, explain = False):
    seq_key_list = list(seq_dict.keys())
    #convert PDB to DSSP file (modernized code from Wei et. al. (2023) DeepProSite) with integration from Biopython PDB.DSSP functionality
    #since we are directly using sequence data from REST API, no need to align structures
    aa_type = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W"] #originally had Y
    SS_type = "HBEGITSC"
    #rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170, 185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    #Default DSSP output (unprocessed): # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
    # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
    # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)

    #QUIET = True -- suppress warnings
    p = PDBParser(QUIET=True)
    if (explain):
        structure = p.get_structure(id=ID, file = "codebase/data/explain-pdb/" + str(ID) + ".pdb")
    else:
        structure = p.get_structure(id=ID, file = "codebase/data/pdb/" + str(ID) + ".pdb") #get PDB
    model = structure[0]
    if (explain):
        dssp = DSSP(model=model, in_file="codebase/data/explain-pdb/" + str(ID) + ".pdb", dssp="mkdssp") #use Biopython & mkdssp to create DSSP dictionary
    else:
        dssp = DSSP(model=model, in_file="codebase/data/pdb/" + str(ID) + ".pdb", dssp="mkdssp") #use Biopython & mkdssp to create DSSP dictionary
    #dssp_dict = {}
    #print(str(dssp.keys()))
    default_dssp = torch.tensor([[0.0,     0.0,     1.0,     1.0,   0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).to(device) #for when AA from sequence cannot be found in PDB, may not be needed now that we are extracting directly from PDB
    #                       sin(phi) sin(psi) cos(phi) cos(psi) rASA  1    2    3    4    5    6    7    8    9
    #for chain_id in seq_key_list:
    dssp_feat = default_dssp.repeat(len(seq_dict[chainid]), 1) #create DSSP feature matrix (n x 14) & save to dict
    #print(np.shape(dssp_feat))
    #dssp_dict[chainid] = dssp_feature
    
    #cnt = 0
    #chain_id = list(dssp.keys())[0][0] #should default to primary chain if working through multiple chains
    #chain_ind_cnt = {}
    #for chain_id in seq_key_list:
    #    chain_ind_cnt[chain_id] = 0
    ind_cnt = 0
    #print(list(dssp.keys()))
    for a_key in list(dssp.keys()):
        chain_id = a_key[0] #this should work as a handle to access dssp_dict
        #print(chain_id)
        if (chain_id != chainid):
            continue
        #print(chain_id)
        try:
            #aa_idx = dssp[a_key][0] - 1 #index of amino acid (-1 to convert from 1-indexing to 0-indexing), from single-chain
            #aa_idx = chain_ind_cnt[chain_id] #index of amino acid with unknowns removed
            aa_idx = ind_cnt
            #print(aa_idx)
            aa = dssp[a_key][1] #retrieve aminoacid ID, if X, assign default DSSP
            SS = dssp[a_key][2]
            if (aa not in aa_type):
                ind_cnt += 1
                continue #can't process nonstandard 20 amino acids
            if SS == "-":
                SS = "C" #none secondary structure, but not unknown
            SS_vec = np.zeros(9) #last dimension represents "unknown" for missing residue (default)
            SS_vec[SS_type.find(SS)] = 1.0
            PHI = float(dssp[a_key][4]) * np.pi / 180 #default value 2pi
            PSI = float(dssp[a_key][5]) * np.pi / 180 #default value 2pi
            rASA = float(dssp[a_key][3]) #default value 0
            aa_feat = torch.cat((torch.tensor([np.sin(PHI), np.sin(PSI), np.cos(PHI), np.cos(PSI), rASA]).to(device), torch.tensor(SS_vec).to(device))) #concat to form DSSP for specific node (CUDA-accelerated) 
            dssp_feat[aa_idx] = aa_feat #update DSSP feature matrix
            ind_cnt += 1
            #cnt += 1
            
        except Exception as e:
            print("!! DSSP Exception: " + str(e))
            ind_cnt += 1 #occasionally buggy due to DSSP software issues
            continue
            #print("exception reached for chain " + chain_id)
            #continue
    #for chain_id in seq_key_list:
    #    print("Chain " + chain_id + ": DSSP successfully calculated for " + str(chain_ind_cnt[chain_id]) + " residues")
    print("Chain " + chainid + ": DSSP successfully calculated for " + str(ind_cnt) + " residues")
    return dssp_feat
    
def seq_to_PLM_embedding(seq_dict):
    #Generate batched per-chain PLM embeddings
    seq_key_list = list(seq_dict.keys())
    #adapted from ProtTrans5 Quickstart Guide
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    # Load model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    #protein sequence preprepared
    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = []
    for i in seq_key_list:
        sequence_examples.append(seq_dict[i])
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

def seq_to_PLM_embedding_low_memory(seq):
    #PLM embeddings using sequence as input
    #adapted from ProtTrans5 Quickstart Guide
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    # Load model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
    sequence_examples = [seq]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:len(prot_seq)]) 
    emb_0 = embedding_repr.last_hidden_state[0,:len(seq)] # shape (len(prot_seq) x 1024)
    return emb_0

    # if you want to derive a single representation (per-protein embedding) for the whole protein
    #emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)

def PDB_to_coord_dict(ID, seq_dict, seq_key_list, seq_key_to_idx, seq_lenptr, explain = False):
    #Extract residue-level coordinate data from PDB files
    coord_dict = {}
    cur_idx = 0
    p = PDBParser(QUIET=True)
    if (explain):
        structure = p.get_structure(id=ID, file = "codebase/data/explain-pdb/" + str(ID) + ".pdb")
    else:
        structure = p.get_structure(id=ID, file = "codebase/data/pdb/" + str(ID) + ".pdb") #get PDB
    model = structure[0]
    residue_to_remove = []
    chain_to_remove = []
    for chain in model:
        if not chain.id in seq_key_list:
            continue
        for residue in chain:
            hetero_flag = residue.id[0].strip()
            if not hetero_flag:
                if(len(residue.resname) != 3):
                    print("couldn't locate " + residue.resname)
                    break 
                try:
                    coord_dict[cur_idx] = residue['CA'].coord
                    cur_idx += 1
                except:
                    continue
    
    for residue in list(residue_to_remove):
        model[residue[0]].detach_child(residue[1])

    for chain in list(chain_to_remove):
        model.detach_child(chain)

    return coord_dict
            
def generate_node_features(ID, chainid, seq_dict, seq_key_list, seq_lenptr, explain = False):
    #generate node feature matrix for specified protein
    #originally developed to process multiple chains -- only one chain per protein is used in PPI database
    
    #generate ProtTrans T5 LLM embedding
    #emb_dict = {}
    #try:
    #emb_dict = seq_to_PLM_embedding(seq_dict[chainid])
    plm_feat = seq_to_PLM_embedding_low_memory(seq_dict[chainid])
    #except Exception as e:
        #likely out of memory, try one at a time
        #print("encoding exception: " + str(e) + ", trying sequence-based encoding")
        #torch.cuda.empty_cache()
        #for seq_key in seq_key_list:
        #    emb_dict[seq_key] = seq_to_PLM_embedding_low_memory(seq_dict[seq_key]) #Likely Bugged
        #    torch.cuda.empty_cache()

    #generate 1DMF from sequence (no longer used in MAPLE-GNN implementation)
    feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD']
    seq_manual_feat_dict = feat_engg_manual_main.extract_prot_seq_1D_manual_feat(root_path="codebase/data processing/helpers/", prot_seq = seq_dict[chainid], feature_type_lst = feature_type_lst, deviceType='cuda:0')
    mf_feat = torch.cat((torch.tensor(seq_manual_feat_dict['AC30']), torch.tensor(seq_manual_feat_dict['PSAAC15']), torch.tensor(seq_manual_feat_dict['ConjointTriad']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_C']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_T']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'])))
    mf_feat = mf_feat.repeat(len(seq_dict[chainid]),1).to(device)

    #mf_dict = {}
    #for chain_id in seq_key_list:
        #print(chain_id)
        #print(seq_dict[chain_id])
        #seq_manual_feat_dict = feat_engg_manual_main.extract_prot_seq_1D_manual_feat(root_path="codebase/data processing/helpers/", prot_seq = seq_dict[chain_id], feature_type_lst = feature_type_lst, deviceType='cuda:0')
        #mf_feat = torch.cat((torch.tensor(seq_manual_feat_dict['AC30']), torch.tensor(seq_manual_feat_dict['PSAAC15']), torch.tensor(seq_manual_feat_dict['ConjointTriad']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_C']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_T']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'])))
        #mf_feat = mf_feat.repeat(len(seq_dict[chain_id]),1).to(device)
        #mf_dict[chain_id] = mf_feat
    
    #generate DSSP embedding using DSSP software
    try:
        dssp_feat = PDB_to_DSSP(ID=ID, chainid = chainid, seq_dict=seq_dict, explain=explain)
    except Exception as e:
        print("DSSP error: " + str(e) + ", assigning default")
        default_dssp = torch.tensor([[0.0,     0.0,     1.0,     1.0,   0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).to(device)
        dssp_feat = default_dssp.repeat(len(seq_dict[chainid]), 1)
    
    #dssp_dict = {}
    #try:
    #    dssp_dict = PDB_to_DSSP(ID=ID,seq_dict=seq_dict, explain=explain)
    #except Exception as e:
    #    print("DSSP error: " + str(e) + ", assigning default")
    #    default_dssp = torch.tensor([[0.0,     0.0,     1.0,     1.0,   0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).to(device)
    #    for chain_id in seq_key_list:
    #        dssp_feature = default_dssp.repeat(len(seq_dict[chain_id]), 1) #create DSSP feature matrix (n x 14) & save to dict
    #        print(np.shape(dssp_feature))
    #        dssp_dict[chain_id] = dssp_feature
    

    #Construct concatenated node feature matrix
    node_feat_matrix = torch.tensor(np.zeros((len(seq_dict[chainid]), 2256))).to(device)
    node_feat_matrix = torch.cat((plm_feat, mf_feat, dssp_feat), dim=1).to(device)

    node_feat_matrix = node_feat_matrix.cpu().numpy()
    #normalize
    for i in range(len(node_feat_matrix)):
        prott5 = node_feat_matrix[i,:1024]
        ac = node_feat_matrix[i, 1024:1234]
        pseaac = node_feat_matrix[i, 1234:1269]
        ct = node_feat_matrix[i, 1269:1612]
        ld = node_feat_matrix[i, 1612:2242]
        dssp = node_feat_matrix[i, 2242:] #don't normalize, already between 0 and 1
        #print(str(np.shape(prott5)) + " " + str(np.shape(ac)) + " " + str(np.shape(pseaac)) + " " + str(np.shape(ct)) + " " + str(np.shape(ld)) + " " + str(np.shape(dssp)))
        #should be 1024 210 35 343 630
        #print(str(np.max(dssp)))
        prott5 = normalize(prott5).copy()
        ac = normalize(ac).copy()
        pseaac = normalize(pseaac).copy()
        ct = normalize(ct).copy()
        ld = normalize(ld).copy()
        node_feat_matrix[i,:] = np.concatenate((prott5, ac, pseaac, ct, ld, dssp))

    #node_feat_matrix = torch.tensor(np.zeros((seq_lenptr[-1], 2256))).to(device)
    #for i in range(len(seq_key_list)):
    #    chain_id = seq_key_list[i]
        #node_feat_dict[chain_id] = torch.cat((emb_dict[chain_id], mf_dict[chain_id], dssp_dict[chain_id]), dim=1).to(device)
    #    node_feat_matrix[seq_lenptr[i]:seq_lenptr[i+1],] = torch.cat((emb_dict[chain_id], mf_dict[chain_id], dssp_dict[chain_id]), dim=1).to(device) #should be working based on how seq-lenptr works
        #print(str(chain_id) + " feature matrix shape: " + str(node_feat_dict[chain_id].shape))
        #node_feat_matrix = torch.cat((node_feat_matrix, node_feat_dict[chain_id]),dim=0)

    return torch.from_numpy(node_feat_matrix)

    #generate Node Feature matrix (n x 2256), concat wrt seq_key_list order; since all features generated per-chain the order should not affect the graph implementation
    
    #node_features = torch.cat((llm_feat,mf_feat, dssp_feat), dim=1)
    #print ("Node Feature Matrix Dimensions:  " + str(node_features.shape))

def generate_edge_features(coord_dict, angstrom_cutoff):
    #Generate edge lists and edge features given PDB coordinate dictionary and angstrom cutoff threshold
    edge_list = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features = torch.empty(0,2).to(device)
    for i in range(len(coord_dict)):
        for j in range(i, len(coord_dict)):
            if (i == j): 
                dist_ij = la.norm(coord_dict[i] - coord_dict[j], ord=2)
                edge_list = torch.cat((edge_list, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features = torch.cat((edge_features, torch.from_numpy(np.array([[(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
            else:
                dist_ij = la.norm(coord_dict[i] - coord_dict[j], ord=2)
                if (dist_ij < angstrom_cutoff):
                    edge_list = torch.cat((edge_list, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features = torch.cat((edge_features, torch.from_numpy(np.array([[(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list = torch.cat((edge_list, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features = torch.cat((edge_features, torch.from_numpy(np.array([[(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)

    return edge_list.T, edge_features

def generate_alternate_edge_features(coord_dict):
    #Generate alternate cutoff edge feature (in bulk & in parallel), (6, 7, 8, 9, 10)

    edge_list_6 = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features_6 = torch.empty(0,2).to(device)
    edge_list_7 = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features_7 = torch.empty(0,2).to(device)
    edge_list_8 = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features_8 = torch.empty(0,2).to(device)
    edge_list_9 = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features_9 = torch.empty(0,2).to(device)
    edge_list_10 = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features_10 = torch.empty(0,2).to(device)
    for i in range(len(coord_dict)):
        for j in range(i, len(coord_dict)):
            if (i == j): 
                dist_ij = la.norm(coord_dict[i] - coord_dict[j], ord=2)
                edge_list_6 = torch.cat((edge_list_6, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features_6 = torch.cat((edge_features_6, torch.from_numpy(np.array([[(6.0 - dist_ij)/(6.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                edge_list_7 = torch.cat((edge_list_7, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features_7 = torch.cat((edge_features_7, torch.from_numpy(np.array([[(7.0 - dist_ij)/(7.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                edge_list_8 = torch.cat((edge_list_8, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features_8 = torch.cat((edge_features_8, torch.from_numpy(np.array([[(8.0 - dist_ij)/(8.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                edge_list_9 = torch.cat((edge_list_9, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features_9 = torch.cat((edge_features_9, torch.from_numpy(np.array([[(9.0 - dist_ij)/(9.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                edge_list_10 = torch.cat((edge_list_10, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features_10 = torch.cat((edge_features_10, torch.from_numpy(np.array([[(10.0 - dist_ij)/(10.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
            else:
                dist_ij = la.norm(coord_dict[i] - coord_dict[j], ord=2)
                if (dist_ij < 6.0):
                    #edge
                    edge_list_6 = torch.cat((edge_list_6, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features_6 = torch.cat((edge_features_6, torch.from_numpy(np.array([[(6.0 - dist_ij)/(6.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list_6 = torch.cat((edge_list_6, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features_6 = torch.cat((edge_features_6, torch.from_numpy(np.array([[(6.0 - dist_ij)/(6.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                if (dist_ij < 7.0):
                    #edge
                    edge_list_7 = torch.cat((edge_list_7, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features_7 = torch.cat((edge_features_7, torch.from_numpy(np.array([[(7.0 - dist_ij)/(7.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list_7 = torch.cat((edge_list_7, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features_7 = torch.cat((edge_features_7, torch.from_numpy(np.array([[(7.0 - dist_ij)/(7.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                if (dist_ij < 8.0):
                    #edge
                    edge_list_8 = torch.cat((edge_list_8, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features_8 = torch.cat((edge_features_8, torch.from_numpy(np.array([[(8.0 - dist_ij)/(8.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list_8 = torch.cat((edge_list_8, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features_8 = torch.cat((edge_features_8, torch.from_numpy(np.array([[(8.0 - dist_ij)/(8.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                if (dist_ij < 9.0):
                    #edge
                    edge_list_9 = torch.cat((edge_list_9, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features_9 = torch.cat((edge_features_9, torch.from_numpy(np.array([[(9.0 - dist_ij)/(9.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list_9 = torch.cat((edge_list_9, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features_9 = torch.cat((edge_features_9, torch.from_numpy(np.array([[(9.0 - dist_ij)/(9.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                if (dist_ij < 10.0):
                    #edge
                    edge_list_10 = torch.cat((edge_list_10, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features_10 = torch.cat((edge_features_10, torch.from_numpy(np.array([[(10.0 - dist_ij)/(10.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list_10 = torch.cat((edge_list_10, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features_10 = torch.cat((edge_features_10, torch.from_numpy(np.array([[(10.0 - dist_ij)/(10.0), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                
    return edge_list_6.T, edge_list_7.T, edge_list_8.T, edge_list_9.T, edge_list_10.T, edge_features_6, edge_features_7, edge_features_8, edge_features_9, edge_features_10
def generate_alternate_edge_features2(coord_dict):
    #Generate alternate cutoff edge feature (in bulk & in parallel), (6.5, 7.5, 8.5, 9.5)

    edge_list_6_5 = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features_6_5 = torch.empty(0,2).to(device)
    edge_list_7_5 = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features_7_5 = torch.empty(0,2).to(device)
    edge_list_8_5 = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features_8_5 = torch.empty(0,2).to(device)
    edge_list_9_5 = torch.empty((0,2), dtype=torch.long).to(device)
    edge_features_9_5 = torch.empty(0,2).to(device)
    for i in range(len(coord_dict)):
        for j in range(i, len(coord_dict)):
            if (i == j): 
                dist_ij = la.norm(coord_dict[i] - coord_dict[j], ord=2)
                edge_list_6_5 = torch.cat((edge_list_6_5, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features_6_5 = torch.cat((edge_features_6_5, torch.from_numpy(np.array([[(6.5 - dist_ij)/(6.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                edge_list_7_5 = torch.cat((edge_list_7_5, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features_7_5 = torch.cat((edge_features_7_5, torch.from_numpy(np.array([[(7.5 - dist_ij)/(7.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                edge_list_8_5 = torch.cat((edge_list_8_5, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features_8_5 = torch.cat((edge_features_8_5, torch.from_numpy(np.array([[(8.5 - dist_ij)/(8.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                edge_list_9_5 = torch.cat((edge_list_9_5, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                edge_features_9_5 = torch.cat((edge_features_9_5, torch.from_numpy(np.array([[(9.5 - dist_ij)/(9.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
            else:
                dist_ij = la.norm(coord_dict[i] - coord_dict[j], ord=2)
                if (dist_ij < 6.5):
                    #edge
                    edge_list_6_5 = torch.cat((edge_list_6_5, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features_6_5 = torch.cat((edge_features_6_5, torch.from_numpy(np.array([[(6.5 - dist_ij)/(6.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list_6_5 = torch.cat((edge_list_6_5, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features_6_5 = torch.cat((edge_features_6_5, torch.from_numpy(np.array([[(6.5 - dist_ij)/(6.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                if (dist_ij < 7.5):
                    #edge
                    edge_list_7_5 = torch.cat((edge_list_7_5, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features_7_5 = torch.cat((edge_features_7_5, torch.from_numpy(np.array([[(7.5 - dist_ij)/(7.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list_7_5 = torch.cat((edge_list_7_5, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features_7_5 = torch.cat((edge_features_7_5, torch.from_numpy(np.array([[(7.5 - dist_ij)/(7.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                if (dist_ij < 8.5):
                    #edge
                    edge_list_8_5 = torch.cat((edge_list_8_5, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features_8_5 = torch.cat((edge_features_8_5, torch.from_numpy(np.array([[(8.5 - dist_ij)/(8.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list_8_5 = torch.cat((edge_list_8_5, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features_8_5 = torch.cat((edge_features_8_5, torch.from_numpy(np.array([[(8.5 - dist_ij)/(8.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                if (dist_ij < 9.5):
                    #edge
                    edge_list_9_5 = torch.cat((edge_list_9_5, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_features_9_5 = torch.cat((edge_features_9_5, torch.from_numpy(np.array([[(9.5 - dist_ij)/(9.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                    edge_list_9_5 = torch.cat((edge_list_9_5, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    edge_features_9_5 = torch.cat((edge_features_9_5, torch.from_numpy(np.array([[(9.5 - dist_ij)/(9.5), np.abs(j-i)/len(coord_dict)]])).to(device)), 0)
                
                
    return edge_list_6_5.T, edge_list_7_5.T, edge_list_8_5.T, edge_list_9_5.T, edge_features_6_5, edge_features_7_5, edge_features_8_5, edge_features_9_5

def save_alternate_edges(ID):
    #save alternate edge features for cutoffs 6, 7, 8, 9, 10
    seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx = ID_to_essentials(ID=str(ID))
    coord_dict = PDB_to_coord_dict(ID=ID, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_key_to_idx = seq_key_to_idx, seq_lenptr=seq_lenptr)
    edge_list_6, edge_list_7, edge_list_8, edge_list_9, edge_list_10, edge_features_6, edge_features_7, edge_features_8, edge_features_9, edge_features_10 = generate_alternate_edge_features(coord_dict)
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_6.npy", edge_list_6.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_7.npy", edge_list_7.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_8.npy", edge_list_8.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_9.npy", edge_list_9.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_10.npy", edge_list_10.cpu().numpy())
    
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_6.npy", edge_features_6.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_7.npy", edge_features_7.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_8.npy", edge_features_8.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_9.npy", edge_features_9.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_10.npy", edge_features_10.cpu().numpy()) 
def save_alternate_edges2(ID):
    #save alternate edge features for cutoffs 6.5, 7.5, 8.5, 9.5
    seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx = ID_to_essentials(ID=str(ID))
    coord_dict = PDB_to_coord_dict(ID=ID, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_key_to_idx = seq_key_to_idx, seq_lenptr=seq_lenptr)
    edge_list_6_5, edge_list_7_5, edge_list_8_5, edge_list_9_5, edge_features_6_5, edge_features_7_5, edge_features_8_5, edge_features_9_5 = generate_alternate_edge_features2(coord_dict)
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_6_5.npy", edge_list_6_5.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_7_5.npy", edge_list_7_5.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_8_5.npy", edge_list_8_5.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_9_5.npy", edge_list_9_5.cpu().numpy())
    
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_6_5.npy", edge_features_6_5.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_7_5.npy", edge_features_7_5.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_8_5.npy", edge_features_8_5.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_9_5.npy", edge_features_9_5.cpu().numpy()) 

def save_features(ID, chainid, seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx):
    #save node and edge features in .npy compressed format to avoid recomputation
    node_feat_matrix = generate_node_features(ID=ID, chainid = chainid, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_lenptr=seq_lenptr)
    np.save("codebase/data/npy/" + str(ID) + "-node_feat.npy", node_feat_matrix.cpu().numpy())
    coord_dict = PDB_to_coord_dict(ID=ID, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_key_to_idx = seq_key_to_idx, seq_lenptr=seq_lenptr)
    for i in range(len(seq_key_list)):
        #i-th chain: seq_lenptr[i] <= x < seq_lenptr[i+1]
        if (seq_key_list[i] != chainid):
            continue
        else:
            coord_list = list(coord_dict.items())[seq_lenptr[i]:seq_lenptr[i+1]]
            for j in range(len(coord_list)):
                coord_list[j] = (coord_list[j][0] - seq_lenptr[i],) + coord_list[j][1:]
            chain_coord_dict = dict(coord_list)
    edge_list_6, edge_list_7, edge_list_8, edge_list_9, edge_list_10, edge_features_6, edge_features_7, edge_features_8, edge_features_9, edge_features_10 = generate_alternate_edge_features(chain_coord_dict)
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_6.npy", edge_list_6.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_7.npy", edge_list_7.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_8.npy", edge_list_8.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_9.npy", edge_list_9.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_10.npy", edge_list_10.cpu().numpy())
    
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_6.npy", edge_features_6.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_7.npy", edge_features_7.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_8.npy", edge_features_8.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_9.npy", edge_features_9.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_10.npy", edge_features_10.cpu().numpy()) 

    edge_list_6_5, edge_list_7_5, edge_list_8_5, edge_list_9_5, edge_features_6_5, edge_features_7_5, edge_features_8_5, edge_features_9_5 = generate_alternate_edge_features2(chain_coord_dict)
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_6_5.npy", edge_list_6_5.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_7_5.npy", edge_list_7_5.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_8_5.npy", edge_list_8_5.cpu().numpy())
    np.save("codebase/data/npy/" + str(ID) + "-edge_list_9_5.npy", edge_list_9_5.cpu().numpy())
    
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_6_5.npy", edge_features_6_5.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_7_5.npy", edge_features_7_5.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_8_5.npy", edge_features_8_5.cpu().numpy()) 
    np.save("codebase/data/npy/" + str(ID) + "-edge_feat_9_5.npy", edge_features_9_5.cpu().numpy()) 
    """
    print(str(np.max(edge_list_7.cpu().numpy())))
    print(str(np.shape(edge_list_7.cpu().numpy())))
    print(str(np.shape(edge_features_7.cpu().numpy())))
    print(str(np.max(edge_list_9_5.cpu().numpy())))
    print(str(np.shape(edge_list_9_5.cpu().numpy())))
    print(str(np.shape(edge_features_9_5.cpu().numpy())))
    """

def normalize(x):
    max = np.max(x)
    min = np.min(x)
    if (abs(max-min) < 1e-7):
        return x-min #reduce overflow, possibility of all zeros
    else:
        return (x-min)/(max-min)

def explain_save_features(ID, chainid, seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx):
    #Save features of specific chains for post-hoc explanation in codebase/data/explain-npy
    node_feat = generate_node_features(ID=ID, chainid = chainid, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_lenptr=seq_lenptr, explain=True).cpu().numpy() #that's so clean
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-node_feat.npy", node_feat)
    node_feat_norm = np.zeros((node_feat.shape[0], 1024+1218+14)) #PLM + 1DMF + DSSP
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
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-node_feat_normalized.npy", node_feat_norm)
    #print("Generating Node Features -- END")
    coord_dict = PDB_to_coord_dict(ID=ID, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_key_to_idx = seq_key_to_idx, seq_lenptr=seq_lenptr, explain=True)
    for i in range(len(seq_key_list)):
        #i-th chain: seq_lenptr[i] <= x < seq_lenptr[i+1]
        if (seq_key_list[i] != chainid):
            continue
        else:
            coord_list = list(coord_dict.items())[seq_lenptr[i]:seq_lenptr[i+1]]
            for j in range(len(coord_list)):
                coord_list[j] = (coord_list[j][0] - seq_lenptr[i],) + coord_list[j][1:]
            chain_coord_dict = dict(coord_list)
    #print("Generating Adjacency Matrix and Edge Features -- START")
    edge_list_6, edge_list_7, edge_list_8, edge_list_9, edge_list_10, edge_features_6, edge_features_7, edge_features_8, edge_features_9, edge_features_10 = generate_alternate_edge_features(chain_coord_dict)
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_list_6.npy", edge_list_6.cpu().numpy())
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_list_7.npy", edge_list_7.cpu().numpy())
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_list_8.npy", edge_list_8.cpu().numpy())
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_list_9.npy", edge_list_9.cpu().numpy())
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_list_10.npy", edge_list_10.cpu().numpy())
    
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_feat_6.npy", edge_features_6.cpu().numpy()) 
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_feat_7.npy", edge_features_7.cpu().numpy()) 
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_feat_8.npy", edge_features_8.cpu().numpy()) 
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_feat_9.npy", edge_features_9.cpu().numpy()) 
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_feat_10.npy", edge_features_10.cpu().numpy()) 

    edge_list_6_5, edge_list_7_5, edge_list_8_5, edge_list_9_5, edge_features_6_5, edge_features_7_5, edge_features_8_5, edge_features_9_5 = generate_alternate_edge_features2(chain_coord_dict)
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_list_6_5.npy", edge_list_6_5.cpu().numpy())
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_list_7_5.npy", edge_list_7_5.cpu().numpy())
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_list_8_5.npy", edge_list_8_5.cpu().numpy())
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_list_9_5.npy", edge_list_9_5.cpu().numpy())
    
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_feat_6_5.npy", edge_features_6_5.cpu().numpy()) 
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_feat_7_5.npy", edge_features_7_5.cpu().numpy()) 
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_feat_8_5.npy", edge_features_8_5.cpu().numpy()) 
    np.save("codebase/data/explain-npy/" + str(ID) + "-" + str(chainid) + "-edge_feat_9_5.npy", edge_features_9_5.cpu().numpy()) 

def ID_to_explain_graph(ID, chainid):
    seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx = ID_to_essentials(ID=ID, chainid=chainid, explain=True)
    explain_save_features(ID=ID, chainid=chainid, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_lenptr=seq_lenptr, seq_key_to_idx=seq_key_to_idx) #codebase/data/npy/ID-node_feat or ID-edge_feat or ID-edge_list

def ID_to_save_graph(ID, chainid, ang_cutoff=9.0):
    seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx = ID_to_essentials(ID=str(ID), chainid=chainid)
    save_features(ID=ID, chainid=chainid, seq_dict=seq_dict, seq_key_list=seq_key_list, seq_lenptr=seq_lenptr, seq_key_to_idx=seq_key_to_idx, ang_cutoff=ang_cutoff) #codebase/data/npy/ID-node_feat or ID-edge_feat or ID-edge_list

def fetch_pdb_file(pdb_id):
    the_url = "https://files.rcsb.org/download/" + pdb_id
    #the_url = "https://files.wwpdb.org/download/" + pdb_id
    page = r.get(the_url)
    pdb_file = page.content.decode('utf-8')
    pdb_file = pdb_file.replace('\\n', '\n')
    return(pdb_file)

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
        if node >= seq_lenptr[cur_comp+1]:
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Protein PDB id & sequence from ID
    #ID = "3i40" #insulin (2-chain, small)
    #ID = "1bfg" #fibroblast growth factor 1 (1-chain, small)
    #ID = "1a3n" #hemoglobin (4-chain, medium)
    #ID = "1tup" #P53 tumor suppressor (3-chain, medium)
    #ID = "3os8" #estrogen receptor (4-chain, large)
    
    #explanation_pairs = [("1TUP", "A"), ("1TUP", "B"), ("6EC0", "A"), ("6EC0", "B"), ("1GRI", "A"), ("1GRI", "B"), ("1BUW", "C"), ("1BUW", "D"), ("3OS8", "A"), ("3OS8", "C"), ("1RFB", "A"), ("1RFB", "B"), ("1AGN", "C"), ("1AGN", "D"), ("5I6Z", "B"), ("5I6Z", "C"), ("7AOS", "A"), ("7AOS", "B"), ("1BJ1", "H"), ("1BJ1", "L"), ("8J7F", "C"), ("8J7F", "D"), ("3I40", "A"), ("3I40", "B"), ("6DDF", "A"), ("6DDF", "R"), ("1IZN", "C"), ("1IZN", "D"), ("1FIN", "A"), ("1FIN", "B"), ("1RUZ", "L"), ("1RUZ", "M"), ("7K43", "B"), ("7K43", "C"), ("2AAI", "A"), ("2AAI", "B"), ("3IFL", "H"), ("3IFL", "L"), ("1BR1", "A"), ("1BR1", "B"), ("3RU8", "H"), ("3RU8", "X")]

    explanation_pairs = [("8QU3", "B"), ("8QU3", "C"), ("1RU0", "A"), ("1RU0", "B")]
    for (ID, chainid) in explanation_pairs:
        ID = ID.lower()
        if (not os.path.exists("codebase/data/explain-pdb/" + str(ID) + ".pdb")):
            #download
            pdbfile = fetch_pdb_file(str(ID) + ".pdb")
            filename = "codebase/data/explain-pdb/" + str(ID) + ".pdb"
            print("Writing " + filename)
            with open(filename, "w") as fd:
                fd.write(pdbfile)
        ID_to_explain_graph(ID.lower(), chainid)
    
    #seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx = ID_to_essentials(ID=str(ID), chainid="A")
    #PDB_to_DSSP(ID, "A", seq_dict)
    #ang_cutoff = 9.0
    #ID_to_save_graph(ID, chainid='A')




    
    
    


