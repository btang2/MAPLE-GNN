import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import sys

from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.insert(0, str(path_root))

from maplegnn.models import MAPLEGNN
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from numpy import linalg as la
import os
from torch_geometric.utils import to_undirected, remove_self_loops
from maplegnn.metrics import get_accuracy, precision, sensitivity, specificity, f_score
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.ticker as ticker
import networkx as nx
import matplotlib.patches as mpatches 
from Bio.PDB import PDBParser
import colorsys

#calculate and visualize MAPLE-GNN node importance attributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def construct_graph(data_dir, graph_id, batch_size, cutoff):
    node_features = np.load(os.path.join(data_dir, f'{graph_id}-node_feat_reduced_dssp.npy')) #normalized or reduced, depending on the model
    edge_indices = np.load(os.path.join(data_dir, f'{graph_id}-edge_list_{cutoff}.npy'))
    edge_features = np.load(os.path.join(data_dir, f'{graph_id}-edge_feat_{cutoff}.npy'))
    #print("node: " + str(np.shape(node_features)) + " edgelist: " + str(np.shape(edge_indices)) + " edgefeat: " + str(np.shape(edge_features)))

    node_features = torch.from_numpy(node_features).float().to(device)
    edge_indices = torch.from_numpy(edge_indices).long().t().T.contiguous().to(device) 
    edge_features = torch.from_numpy(edge_features).float().to(device)

    return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_features, batch_size = batch_size, graph_id = graph_id)

def extract_scores(model, modelname, p1_ID, p2_ID, label):
    #extract node scores
    #in progress
    # load the test data
    data_dir = 'codebase/data/explain-npy/'
    model_path = "codebase/model_instances/" + str(modelname) + ".pth"
    batch_size = 1
    protein1 = construct_graph(data_dir, p1_ID, batch_size)
    print(protein1)
    protein2 = construct_graph(data_dir, p2_ID, batch_size)

    model = model()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    # Get model prediction
    model.eval()
    with torch.no_grad():
        out, p1_perm, p1_attr, p2_perm, p2_attr = model(protein1, protein2, train=False)
        print(str(modelname) + " Prediction:" + str(out))

    #p1_attn_norm = (p1_attn - torch.mean(p1_attn) / torch.var(p1_attn)) not applicable I think
    #p2_attn_norm = (p2_attn - torch.mean(p2_attn) / torch.var(p2_attn))
    # Get attention scores
    return out.item(), p1_perm, p1_attr, p2_perm, p2_attr

def ID_to_essentials(ID):
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    seq_dict = {}
    p = PDBParser(QUIET=True)
    structure = p.get_structure(id=ID, file = "codebase/data/explain-pdb/" + str(ID) + ".pdb")
    model = structure[0]
    residue_to_remove = []
    chain_to_remove = []
    for chain in model:
            #print("found the chain!")
        seq = ""
        for residue in chain:
            hetero_flag = residue.id[0].strip()
            if not hetero_flag:
                if (residue.resname in d3to1.keys()):
                    seq += d3to1[residue.resname]
                elif(len(residue.resname) != 3):
                    break #not a protein chain, Bugged?
                    #donothing = True
                else:
                    seq += 'X'
        if(len(seq) <= 1):
            chain_to_remove.append(chain.id) #chain of hetatm's/extraneous chain (remove, required for 1DMF to function properly)
        else:
            #print(len(seq))
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
    
    return seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx

def PDB_to_coord_dict(ID, seq_dict, seq_key_list, seq_key_to_idx, seq_lenptr):
    #first, convert PDB to list of coords
    #how to "average" coords of atoms to obtain coord of residue? use alpha-carbon
    coord_dict = {}
    cur_idx = 0
    #print("Extracting Chain Information -- START")
    p = PDBParser(QUIET=True)
    structure = p.get_structure(id=ID, file = "codebase/data/explain-pdb/" + str(ID) + ".pdb")
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
                    break #not a protein chain, maybe bugged?
                try:
                    coord_dict[cur_idx] = residue['CA'].coord
                    cur_idx += 1
                except:
                    #no Carbon Atom??? probably not an amino acid, continue for now ?
                    continue
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    num_atms = 0
                    for atom in residue:
                        x += atom.coord[0]
                        y += atom.coord[1]
                        z += atom.coord[2]
                        num_atms += 1
                    print("NO ALPHA-CARBON at index " + str(cur_idx) + ". Residue name: " + str(residue.resname) + ", averaging")
                    if (num_atms == 0):
                        print("ERROR: trying to process residue with no atoms")
                    coord_dict[cur_idx] = np.array([x/num_atms, y/num_atms, z/num_atms])
                
        

    for residue in list(residue_to_remove):
        model[residue[0]].detach_child(residue[1])

    for chain in list(chain_to_remove):
        model.detach_child(chain)

    
    #print("Extracting Chain Information -- END")
    return coord_dict

def save_PDB_edge_list(ID, chain_1, chain_2, cutoff):
    seq_dict, seq_key_list, seq_lenptr, seq_key_to_idx = ID_to_essentials(ID)
    coord_dict = PDB_to_coord_dict(ID, seq_dict, seq_key_list, seq_key_to_idx, seq_lenptr)
    chain_coord_list = []
    for i in range(len(seq_key_list)): #first pass w/ chain 1
        #i-th chain: seq_lenptr[i] <= x < seq_lenptr[i+1]
        if (seq_key_list[i] == chain_1):
            chain_coord_list = list(coord_dict.items())[seq_lenptr[i]:seq_lenptr[i+1]]
            for j in range(len(chain_coord_list)):
                chain_coord_list[j] = (chain_coord_list[j][0] - seq_lenptr[i],) + chain_coord_list[j][1:]
    baseline = len(chain_coord_list)
    for k in range(len(seq_key_list)): #second pass w/ chain 2
        if (seq_key_list[k] == chain_2):
            chain_2_coord_list = list(coord_dict.items())[seq_lenptr[k]:seq_lenptr[k+1]]
            for m in range(len(chain_2_coord_list)):
                chain_coord_list.append((chain_2_coord_list[m][0] - seq_lenptr[k]+baseline,) + chain_2_coord_list[m][1:])
    #print(len(chain_coord_list))
    new_coord_dict = dict(chain_coord_list)
    
    #print(new_coord_dict.keys())

    edge_list = torch.empty((0,2), dtype=torch.long).to(device)
    for i in range(len(new_coord_dict)):
        for j in range(i, len(new_coord_dict)):
            if (i == j): 
                dist_ij = la.norm(new_coord_dict[i] - new_coord_dict[j], ord=2)
                edge_list = torch.cat((edge_list, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                #adj_matrix[i][j] = 1 #i=j, just process once
                #edge_features[i][j] = torch.tensor([(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]).to(device)
            else:
                dist_ij = la.norm(new_coord_dict[i] - new_coord_dict[j], ord=2)
                if (dist_ij < cutoff):
                    #edge
                    edge_list = torch.cat((edge_list, torch.from_numpy(np.array([[i,j]])).to(device)),0)
                    edge_list = torch.cat((edge_list, torch.from_numpy(np.array([[j,i]])).to(device)),0)
                    #adj_matrix[i][j] = 1
                    #adj_matrix[j][i] = 1
                    #edge_features[i][j] = torch.tensor([(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]).to(device)
                    #edge_features[i][j] = torch.tensor([(angstrom_cutoff - dist_ij)/(angstrom_cutoff), np.abs(j-i)/len(coord_dict)]).to(device)
                else:
                    #no edge
                    no_edge = True
                    #edge_features[i][j] = torch.tensor([0,0]).to(device)
                    #edge_features[j][i] = torch.tensor([0,0]).to(device)
    np.save(f'codebase/data/explain-npy/{ID.upper()}-combined_edge_list_{cutoff}.npy', edge_list.T.cpu().numpy())
    new_seq_lenptr = np.array([0, baseline, len(chain_coord_list)])
    print(new_seq_lenptr)
    new_seq_dict = {}
    new_seq_dict[chain_1] = seq_dict[chain_1]
    new_seq_dict[chain_2] = seq_dict[chain_2]
    new_seq_key_list = [chain_1, chain_2]
    return new_seq_dict, new_seq_key_list, new_seq_lenptr

def integrated_gradients(model, modelname, p1_ID, p2_ID, cutoff, steps=100):
    data_dir = 'codebase/data/explain-npy/'
    model_path = "codebase/model_instances/" + str(modelname) + ".pth"
    batch_size = 1
    p1_data = construct_graph(data_dir, p1_ID, batch_size, cutoff)
    p2_data = construct_graph(data_dir, p2_ID, batch_size, cutoff)

    model = model()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    output = model(p1_data, p2_data)
    print("Prediction: " + str(output))

    p1_baseline = torch.zeros_like(p1_data.x)

    p1_scaled_inputs = [p1_baseline + (float(i) / steps) * (p1_data.x - p1_baseline) for i in range(steps + 1)]
    p1_grads = []

    for scaled_input in p1_scaled_inputs:
        p1_scaled_data = Data(x=scaled_input, edge_index=p1_data.edge_index, edge_attr=p1_data.edge_attr)
        p1_scaled_data = p1_scaled_data.to(device)
        model.eval()
        p1_scaled_input = p1_scaled_data.x
        p1_scaled_input.requires_grad = True
        p1_output = model(p1_scaled_data, p2_data)
        p1_loss = p1_output[0, 0]
        p1_loss.backward()
        p1_grads.append(p1_scaled_input.grad.data.cpu().numpy())

    p1_avg_grads = np.mean(p1_grads, axis=0)
    p1_integrated_grads = (p1_data.x.cpu().data.numpy() - p1_baseline.cpu().data.numpy()) * p1_avg_grads
    
    p2_baseline = torch.zeros_like(p2_data.x)

    p2_scaled_inputs = [p2_baseline + (float(i) / steps) * (p2_data.x - p2_baseline) for i in range(steps + 1)]
    p2_grads = []

    for scaled_input in p2_scaled_inputs:
        p2_scaled_data = Data(x=scaled_input, edge_index=p2_data.edge_index, edge_attr=p2_data.edge_attr)
        p2_scaled_data = p2_scaled_data.to(device)
        model.eval()
        p2_scaled_input = p2_scaled_data.x
        p2_scaled_input.requires_grad = True
        p2_output = model(p1_data, p2_scaled_data)
        p2_loss = p2_output[0, 0]
        p2_loss.backward()
        p2_grads.append(scaled_input.grad.data.cpu().numpy())

    p2_avg_grads = np.mean(p2_grads, axis=0)
    p2_integrated_grads = (p2_data.x.cpu().data.numpy() - p2_baseline.cpu().data.numpy()) * p2_avg_grads
    

    return p1_integrated_grads, p2_integrated_grads

def calculate_node_importance_from_integrated_grads(integrated_grads):
    # Sum the integrated gradients across all features for each node
    node_importance = np.sum(integrated_grads, axis=1)
    return node_importance

def calculate_feature_importance_from_integrated_grads(integrated_grads):
    # Sum the integrated gradients across all nodes for each feature
    feature_importance = np.sum(integrated_grads, axis=0)
    return feature_importance

def calculate_IG_gt_vectors(model, modelname, p1_ID, p2_ID, p1_gt, p2_gt, cutoff):
    p1_ig, p2_ig = integrated_gradients(model, modelname, p1_ID, p2_ID, cutoff)
    p1_node_importance = calculate_node_importance_from_integrated_grads(p1_ig)
    p2_node_importance = calculate_node_importance_from_integrated_grads(p2_ig)
    #p1_node_norm = p1_node_importance
    p1_node_norm = (p1_node_importance - np.mean(p1_node_importance))/np.std(p1_node_importance)
    #p2_node_norm = p2_node_importance 
    p2_node_norm = (p2_node_importance - np.mean(p2_node_importance))/np.std(p2_node_importance)
    p1_gt_transform = np.zeros(len(p1_node_importance), dtype=int)
    p2_gt_transform = np.zeros(len(p2_node_importance), dtype=int)

    for i in range(len(p1_gt)):
        p1_gt_transform[p1_gt[i] - 1] = 1
    for j in range(len(p2_gt)):
        p2_gt_transform[p2_gt[j] - 1] = 1

    print(f"{p1_ID} explanation attribution sum: {np.sum(np.dot(p1_node_norm, p1_gt_transform))}")
    print(f"{p2_ID} explanation attribution sum: {np.sum(np.dot(p2_node_norm, p2_gt_transform))}")
    calculate_metrics(p1_ID, p2_ID, p1_node_norm, p2_node_norm, p1_gt_transform, p2_gt_transform)

    return p1_node_norm, p2_node_norm, p1_gt_transform, p2_gt_transform

def calculate_metrics(p1_ID, p2_ID, p1_node_importance, p2_node_importance, p1_gt_transform, p2_gt_transform):
    
    p1_acc = get_accuracy(p1_gt_transform, p1_node_importance, 0.0)
    p1_prec = precision(p1_gt_transform, p1_node_importance, 0.0)
    p1_sens = sensitivity(p1_gt_transform, p1_node_importance, 0.0)
    p1_spec = specificity(p1_gt_transform, p1_node_importance, 0.0)
    p1_f1 = f_score(p1_gt_transform, p1_node_importance, 0.0)

    print(f"##### Protein 1 ({p1_ID}) explanation performance: ")
    print(f"Accuracy: {p1_acc}")
    print(f"Precision: {p1_prec}")
    print(f"Sensitivity: {p1_sens}")
    print(f"Specificity: {p1_spec}")
    print(f"F-Score: {p1_f1}")

    print("#####")

    p2_acc = get_accuracy(p2_gt_transform, p2_node_importance, 0.0)
    p2_prec = precision(p2_gt_transform, p2_node_importance, 0.0)
    p2_sens = sensitivity(p2_gt_transform, p2_node_importance, 0.0)
    p2_spec = specificity(p2_gt_transform, p2_node_importance, 0.0)
    p2_f1 = f_score(p2_gt_transform, p2_node_importance, 0.0)

    print(f"##### Protein 2 ({p2_ID}) explanation performance: ")
    print(f"Accuracy: {p2_acc}")
    print(f"Precision: {p2_prec}")
    print(f"Sensitivity: {p2_sens}")
    print(f"Specificity: {p2_spec}")
    print(f"F-Score: {p2_f1}")

    print("#####")


def visualize_1d_IG(model, modelname, p1_ID, p2_ID, p1_node_norm, p2_node_norm, p1_gt, p2_gt, cutoff):
    #p1_node_norm, p2_node_norm, p1_gt_transform, p2_gt_transform = calculate_IG_gt_vectors(model, modelname, p1_ID, p2_ID, p1_gt, p2_gt, cutoff)

    #graphing
    p1_x = np.arange(1, len(p1_node_norm)+1)
    p2_x = np.arange(1, len(p2_node_norm)+1)
    p1_colors = []
    p2_colors = []
    for k in range(len(p1_node_norm)):
            if (p1_node_norm[k] > 0):
                p1_colors.append('tab:blue')
            else:
                p1_colors.append('tab:gray')
        
    for l in range(len(p2_node_norm)):
        if (p2_node_norm[l] > 0):
            p2_colors.append('tab:orange')
        else:
            p2_colors.append('tab:gray')
   
    
    sns.set_theme('paper')
    sns.set_style('ticks')
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, dpi=1200)
    f.set_figwidth(10)
    f.set_figheight(3)
    
    # ax1.plot(df.x, df.y)
    ax1.set_title(f'{p1_ID} Residue Attributions',fontsize=10)
    ax1.set_ylabel("Attribution Score", fontsize=9, weight='bold')
    ax1.set_xlabel("Residue Index",fontsize=9, weight='bold')
    #ax1.get_xaxis().set_visible(False)
    ax1.set_xticks(range(0,len(p1_x), 1))
    #start, end = ax1.get_ylim()
    #ax1.yaxis.set_ticks(np.arange(start, end, 1.0))

    sns.barplot(x=p1_x, y=p1_node_norm, palette=p1_colors, ax=ax1, hue=p1_x, legend=False)
    for i in p1_gt:
        ax1.axvspan(i-1.5, i-0.5, color='tab:red', alpha=0.1, linewidth=0)
    ax1.tick_params(axis='x', labelsize=7)
    ax1.tick_params(axis='y', labelsize=7)

    ax2.yaxis.set_tick_params(labelleft=True)
    ax2.set_title(f'{p2_ID} Residue Attributions', fontsize=10)
    ax2.set_ylabel("Attribution Score", fontsize=9, weight = 'bold')
    ax2.set_xlabel("Residue Index", fontsize=9, weight = 'bold')
    #ax2.get_xaxis().set_visible(False)
    ax2.set_xticks(range(0,len(p2_x), 1))
    sns.barplot(x=p2_x, y=p2_node_norm, palette=p2_colors, ax=ax2, hue=p2_x, legend=False)
    for j in p2_gt:
        ax2.axvspan(j-1.5, j-0.5, color='tab:red', alpha=0.1, linewidth=0)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.tick_params(axis='y', labelsize=7)
    plt.suptitle("Human Insulin Protein Binding Interface (3I40-A/B) Prediction Attribution Scores", fontsize = 12, weight='bold') #change with ideal protein
    plt.tight_layout()
    plt.savefig(f"codebase/images/{p1_ID[:4]}-attribution.png", dpi=300)
    plt.close()

def visualize_2d_graph_IG(ID, chain_1, chain_2, c1_norm_attr, c2_norm_attr, c1_gt_transform, c2_gt_transform, cutoff):
    #Here, c1_attr/c2_attr represents attribution scores (MAPLE-GNN normalized integrated gradients attributions or ground truth)
    #run calculate_IG_gt_vectors beforehand to use as input
    c1_max_attr = np.max(c1_norm_attr) 
    c2_max_attr = np.max(c2_norm_attr)
    seq_dict, seq_key_list, seq_lenptr = save_PDB_edge_list(ID, chain_1, chain_2, cutoff)
    edgelist = np.load(os.path.join(f'codebase/data/explain-npy/', f'{ID}-combined_edge_list_{cutoff}.npy')).T

    f, (ax1, ax2) = plt.subplots(1, 2, dpi=1200)
    f.set_figwidth(10)
    f.set_figheight(3)
    H = nx.from_edgelist(edgelist) #convert to CPU for graph visualization
    K = nx.Graph() #sprted
    K.add_nodes_from(sorted(H.nodes(data=True)))
    K.add_edges_from(H.edges(data=True))
    K.remove_edges_from(nx.selfloop_edges(K)) #remove self-loops for better visualizations (self-loops implied)
    colors = ["#1f77b4", "#ff7f0e"] #just two colors for two chains; tab:blue/tab:orange
    color_map_ig = []
    color_map_gt = []
    color_map_original = []
    cur_comp = 0
    relabel_dict = {}
    for node in K:
        
        if node >= seq_lenptr[cur_comp+1]:
            cur_comp += 1
        if cur_comp == 0:
            relabel_dict[node] = str(chain_1) + str(node + 1) # 1-indexing
        if cur_comp == 1:
            relabel_dict[node] = str(chain_2) + str(node + 1 - len(c1_norm_attr)) 
        #scale colors based on attribution score
        rgb = mpl.colors.ColorConverter.to_rgb(colors[cur_comp])
        h, l, s = colorsys.rgb_to_hls(*rgb)
        original_color = mpl.colors.rgb2hex(colorsys.hls_to_rgb(h,l,0.8))
        l_gt = l
        s_gt = s
        if (cur_comp == 0):
            if (c1_norm_attr[node] <= 0.0):
                s = 0.0
                l = 0.8 #contrast non-active nodes from chains
            else:
                s = 0.4 + 0.6*(c1_norm_attr[node] / c1_max_attr * s)
            if (c1_gt_transform[node] <= 0.0):
                s_gt = 0.0
                l_gt = 0.8
            else: 
                s_gt = 1.0
        elif (cur_comp == 1):
            if (c2_norm_attr[node-len(c1_norm_attr)] <= 0.0):
                s = 0.0
                l = 0.6
            else:
                s = 0.4 + 0.6*(c2_norm_attr[node-len(c1_norm_attr)] / c2_max_attr * s)
            if (c2_gt_transform[node-len(c1_norm_attr)] <= 0.0):
                s_gt = 0.0
                l_gt = 0.6
            else: 
                s_gt = 1.0
        scaled_color_ig = mpl.colors.rgb2hex(colorsys.hls_to_rgb(h,l,s))
        scaled_color_gt = mpl.colors.rgb2hex(colorsys.hls_to_rgb(h,l_gt,s_gt))
        color_map_ig.append(scaled_color_ig)
        color_map_gt.append(scaled_color_gt)
        color_map_original.append(original_color)
    orig_patch_arr = []
    patch_arr = []
    for i in range(len(seq_key_list)):
        patch_arr.append(mpatches.Patch(color=colors[i], label='Chain ' + str(seq_key_list[i]) + " Interface"))
        orig_patch_arr.append(mpatches.Patch(color=colors[i], label='Chain ' + str(seq_key_list[i])))

    ax1.legend(handles=patch_arr,loc=1, prop={'size':6, 'weight': 'bold'})
    ax2.legend(handles=patch_arr,loc=1, prop={'size':6, 'weight': 'bold'})
    #ax3.legend(handles=orig_patch_arr,loc=1, prop={'size':6})

    #relabel to 1-index & by chain
    G = nx.relabel.relabel_nodes(K, relabel_dict)
    pos = nx.spring_layout(G,seed=1)
    n_nodes = G.number_of_nodes()
    #draw graph based on overall size: small (<=250), medium (>250, <= 750), large (>750) amino acids
    #average human protein is <500 so would fall into small or medium category
    width = 1
    node_size = 100
    font_size = 3
    
    if (n_nodes <= 250):
        #small
        width = 0.4
        node_size = 150
        font_size = 4
        #node_size = 800
        #font_size = 14
    elif (n_nodes > 250 and n_nodes <= 750):
        width = 0.2 #update 
        node_size = 50
        font_size = 2
    else:
        #n_nodes > 750
        width = 0.1
        node_size = 20
        font_size = 1
    
    nx.draw_networkx_edges(G=G, pos=pos,width=width, ax=ax1)
    nx.draw_networkx_nodes(G=G, pos=pos, node_color = color_map_ig, node_size=node_size, ax=ax1) #color_map
    nx.draw_networkx_labels(G=G, pos=pos, font_size = font_size, font_weight= 'bold', ax=ax1)
    ax1.set_title(f"MAPLE-GNN Binding Interface Prediction", fontsize = 10)

    nx.draw_networkx_edges(G=G, pos=pos,width=width, ax=ax2)
    nx.draw_networkx_nodes(G=G, pos=pos, node_color = color_map_gt, node_size=node_size, ax=ax2) #color_map
    nx.draw_networkx_labels(G=G, pos=pos, font_size = font_size, font_weight='bold', ax=ax2)
    ax2.set_title(f"PDBePISA Ground Truth Annotation", fontsize = 10)

    #nx.draw_networkx_edges(G=G, pos=pos,width=width, ax=ax3)
    #nx.draw_networkx_nodes(G=G, pos=pos, node_color = color_map_original, node_size=node_size, ax=ax3) #color_map
    #nx.draw_networkx_labels(G=G, pos=pos, font_size = font_size, ax=ax3)
    #ax3.set_title(f"Baseline Protein Graph Representation", fontsize = 10)
    
    strFile = "codebase/images/" + str(ID) + "-combined.png"

     #remove margin box
    plt.suptitle("Human Insulin (3I40-A/B) Graph-Level Binding Interface Attribution Comparison", fontsize=12, weight = 'bold')
    #              Human Insulin Protein Binding Interface (3I40-A/B) Prediction Attribution Scores With Integrated Gradients
    plt.tight_layout()
    plt.savefig(strFile, dpi=300)
    plt.close()
    


model = MAPLEGNN
modelname = "CROSSVALMODEL-4" 
cutoff = "9"
p1_ID = "3I40-A"
p2_ID = "3I40-B"
p1_gt = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21]
p2_gt = [2, 3, 4, 5, 6, 7, 8, 11, 14, 15, 18, 19, 22, 23, 24, 25, 26, 27, 28, 30]

c1_norm_attr, c2_norm_attr, c1_gt_transform, c2_gt_transform = calculate_IG_gt_vectors(model, modelname, p1_ID, p2_ID, p1_gt, p2_gt, cutoff)
visualize_1d_IG(model, modelname, p1_ID, p2_ID, c1_norm_attr, c2_norm_attr, p1_gt, p2_gt, cutoff)
visualize_2d_graph_IG(p1_ID[:4], p1_ID[5], p2_ID[5], c1_norm_attr, c2_norm_attr, c1_gt_transform, c2_gt_transform, 9)
