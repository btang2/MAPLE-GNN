#OPEN-SOURCE CODE taken from existing codebase at https://github.com/ShubhrangshuGhosh2000/mat_p2ip_prj

# this is the main module for manual feature extraction/engineering
import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from helpers.AutoCovariance import AutoCovariance
from helpers.ConjointTriad import ConjointTriad
from helpers.LDCTD import LDCTD
from helpers.PSEAAC import PSEAAC

import numpy as np

import time

def extract_prot_seq_1D_manual_feat(root_path='./', prot_seq = 'MIFTPFLPPADLSVFQNVKGLQNDPE', feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD'], deviceType='cpu'):
    #print('[Extracting 1D Manual Features -- START]')
    fastas = [('999', prot_seq)]
    featureSets = set(feature_type_lst)
    # the dictionary to be returned
    seq_manual_feat_dict = {}

    if 'AC30' in featureSets:
        #print("Calculating 'AC30' feature - Start") for debugging
        ac = AutoCovariance(fastas, lag=30, deviceType=deviceType)    
        ac30_feat = ac[1][1:]
        for i in range(len(ac30_feat)):
            if (np.isnan(ac30_feat[i])):
                ac30_feat[i] = 0.0 #replace NaN values with 0.0 for sequences with less than 30 length
        seq_manual_feat_dict['AC30'] = ac30_feat
        #print("Calculating 'AC30' feature - End") for debugging

    if 'PSAAC15' in featureSets or 'PSEAAC15' in featureSets:
        #print("Calculating 'PSAAC15' feature - Start") for debugging
        paac = PSEAAC(fastas,lag=15)
        paac_feat = paac[1][1:]
        for i in range(len(paac_feat)):
            if (np.isnan(paac_feat[i])):
                paac_feat[i] = 0.0 #replace NaN values with 0.0 
        seq_manual_feat_dict['PSAAC15'] = paac_feat
        #print("Calculating 'PSAAC15' feature - End") for debugging

    if 'ConjointTriad' in featureSets or 'CT' in featureSets:
        #print("Calculating 'ConjointTriad' feature - Start") for debugging
        ct = ConjointTriad(fastas,deviceType=deviceType)
        ct_feat = ct[1][1:]
        for i in range(len(ct_feat)):
            if (np.isnan(ct_feat[i])):
                ct_feat[i] = 0.0 #replace NaN values with 0.0 
        seq_manual_feat_dict['ConjointTriad'] = ct_feat
        #print("Calculating 'ConjointTriad' feature - End") for debugging

    if 'LD10_CTD' in featureSets:
        #print("Calculating 'LD10_CTD' feature - Start") for debugging
        (comp, tran, dist) = LDCTD(fastas)
        comp_feat = comp[1][1:]
        for i in range(len(comp_feat)):
            if (np.isnan(comp_feat[i])):
                comp_feat[i] = 0.0 #replace NaN values with 0.0
        seq_manual_feat_dict['LD10_CTD_ConjointTriad_C'] = comp_feat
        tran_feat = tran[1][1:]
        for j in range(len(tran_feat)):
            if (np.isnan(tran_feat[j])):
                tran_feat[j] = 0.0 #replace NaN values with 0.0
        seq_manual_feat_dict['LD10_CTD_ConjointTriad_T'] = tran_feat
        dist_feat = dist[1][1:]
        for i in range(len(dist_feat)):
            if (np.isnan(dist_feat[i])):
                dist_feat[i] = 0.0 #replace NaN values with 0.0
        seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'] = dist_feat
        #print("Calculating 'LD10_CTD' feature - End") for debugging

    #print('[Extracting 1D Manual Features -- END]')
    return seq_manual_feat_dict


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    

    #prot_seq = 'SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENL'
    prot_seq = "X" #try to break 1dmf (looks like its more robust now)
    feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD'] #
    seq_manual_feat_dict = extract_prot_seq_1D_manual_feat(root_path, prot_seq = prot_seq, feature_type_lst = feature_type_lst, deviceType='cpu')
    #mf_vector = np.concatenate((seq_manual_feat_dict['AC30'], seq_manual_feat_dict['PSAAC15'], seq_manual_feat_dict['ConjointTriad'], seq_manual_feat_dict['LD10_CTD_ConjointTriad_C'], seq_manual_feat_dict['LD10_CTD_ConjointTriad_T'], seq_manual_feat_dict['LD10_CTD_ConjointTriad_D']))
    print(seq_manual_feat_dict['LD10_CTD_ConjointTriad_D']) 

