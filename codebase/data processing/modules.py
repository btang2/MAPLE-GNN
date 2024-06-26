import os
import sys
from Bio.PDB import PDBParser
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB import PDBList
import numpy as np
import requests as r
import warnings
from transformers import T5EncoderModel, T5Tokenizer
import torch
import re
import time
from helpers import feat_engg_manual_main

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#from pathlib import Path
#path_root = Path(__file__).parents[1]  # upto 'codebase' folder
#sys.path.insert(0, str(path_root))
# print(sys.path)

def PDB_to_DSSP(ID, prot_seq):
    print("[Calculating DSSP -- START]")
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
    dssp = DSSP(model=model, in_file="codebase/data/pdb/" + str(ID) + ".pdb", dssp="mkdssp") #use Biopython & mkdsssp to create DSSP dictionary, this model should only have one chain
    #print(str(dssp.keys()))
    default_dssp = torch.tensor([[0.0,     0.0,     1.0,     1.0,   0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).to(device) #for when AA from sequence cannot be found in PDB
    #                       sin(phi) sin(psi) cos(phi) cos(psi) rASA  1    2    3    4    5    6    7    8    9
    dssp_feature = default_dssp.repeat(len(prot_seq), 1) #create DSSP feature matrix (n x 14)
    #print(np.shape(dssp_feature))
    cnt = 0
    #print("".join(["can calculate dssp for ", str(len(list(dssp.keys()))), " of ", str(len(prot_seq)), " keys"]))
    chain_id = list(dssp.keys())[0][0] #should default to primary chain if working through multiple chains
    for a_key in list(dssp.keys()):
        if (a_key[0] != chain_id): #method attempting to process multiple chains
            break #don't allow method to process multiple chains
        try:
            aa_idx = dssp[a_key][0] - 1 #index of amino acid (-1 to convert from 1-indexing to 0-indexing)
            #aa = dssp[a_key][1] #retrieve aminoacid ID, legacy, no longer needed
            SS = dssp[a_key][2]
            if SS == "-":
                SS = "C" #none secondary structure, but not unknown
            SS_vec = np.zeros(9) #last dimension represents "unknown" for missing residue (default)
            SS_vec[SS_type.find(SS)] = 1.0
            PHI = float(dssp[a_key][4]) * np.pi / 180 #default value 2pi
            PSI = float(dssp[a_key][5]) * np.pi / 180 #default value 2pi
            rASA = float(dssp[a_key][3]) #default value 0
            aa_feat = torch.cat((torch.tensor([np.sin(PHI), np.sin(PSI), np.cos(PHI), np.cos(PSI), rASA]).to(device), torch.tensor(SS_vec).to(device))) #concat to form DSSP for specific node (CUDA-accelerated) 
            dssp_feature[aa_idx] = aa_feat #update DSSP feature matrix
            cnt += 1
            
        except:
            continue
        
    print("DSSP successfully calculated for " + str(cnt) + " of " + str(len(prot_seq)) + " keys")
    print("[Calculating DSSP -- END]")
    return dssp_feature
    
def PDB_to_PLM_embedding(prot_seq):
    print("[Converting Sequence to LLM Embedding -- START]")
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
    sequence_examples.append(prot_seq)
    #print(sequence_examples)
    print(len(sequence_examples[0]))
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:len(prot_seq)]) 
    emb_0 = embedding_repr.last_hidden_state[0,:len(prot_seq)] # shape (len(prot_seq) x 1024)
    print("[Converting Sequence to LLM Embedding -- END]")
    return emb_0

    # if you want to derive a single representation (per-protein embedding) for the whole protein
    #emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)

def ID_to_seq(ID):
    time.sleep(0.01) #to not overload REST API
    data = r.get(f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{str(ID)}').json()[ID.lower()] #query REST for PDB sequence API
    return data[0]['sequence']
    # with open("codebase/data/fasta/" + str(ID) + ".fasta") as handle:
    #    for values in SimpleFastaParser(handle):
    #        return values[1]


if __name__ == '__main__':
    #get_dssp(ID="1tup", ref_seq="SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNT")
    #print(PDB_to_DSSP("1tup", "codebase/data/pdb/1tup.pdb"))
    
    #pdbfile = 'codebase/data/pdb/' + ID + '.pdb'
    #with open(pdbfile) as handle:
    #    sequence = next(SeqIO.parse(handle, "pdb-atom"))
    #with open("codebase/data/fasta/" + ID + ".fasta", "w") as output_handle:
    #    SeqIO.write(sequence, output_handle, "fasta")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Protein PDB id & sequence from ID
    ID = "6bgt"
    seq = ID_to_seq(ID) #eventually make into dict
    #generate ProtTrans T5 LLM embedding
    llm_feat = PDB_to_PLM_embedding(seq).to(device)
    #print(llm_feat.shape)
    #generate 1DMF from sequence
    #possible bug/confounder: AC30 generates NaN values, most likely bug with original fasta file as I cannot seem to reproduce it
    feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD']
    seq_manual_feat_dict = feat_engg_manual_main.extract_prot_seq_1D_manual_feat(root_path="helpers/", prot_seq = seq, feature_type_lst = feature_type_lst, deviceType='cuda:0')
    mf_feat = torch.cat((torch.tensor(seq_manual_feat_dict['AC30']), torch.tensor(seq_manual_feat_dict['PSAAC15']), torch.tensor(seq_manual_feat_dict['ConjointTriad']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_C']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_T']), torch.tensor(seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'])))
    mf_feat = mf_feat.repeat(len(seq),1).to(device)
    #print(mf_feat.shape)
    
    dssp_feat = PDB_to_DSSP(ID=ID, prot_seq=seq).to(device)
    #print(dssp_feat.shape)
    #generate Node Feature matrix (n x 2256)
    node_features = torch.cat((llm_feat,mf_feat, dssp_feat), dim=1)
    print ("Node Feature Matrix Dimensions:  " + str(node_features.shape))


