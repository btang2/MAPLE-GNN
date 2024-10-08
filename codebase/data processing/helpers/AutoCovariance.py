#OPEN-SOURCE CODE taken from existing codebase at https://github.com/ShubhrangshuGhosh2000/mat_p2ip_prj


import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from helpers.Covariance import Covariance


#for default AAIndex IDs, using close approximatation of Guo's calculations

#Guo AAC calculation
#aaIndexParameters = []
#aaIndexParameters.append('GUO_H1') #hydrophobicity, Added Guo's values to AAIndex.  Cannot match these to values in AAIndex.
#aaIndexParameters.append('HOPT810101') #hydrophicility  (Guo listed 1 value as 2, but in aaindex it is 0.2.  Assuming Guo mistyped, but this could affect papers that copied values from their supplementary data)
#aaIndexParameters.append('KRIW790103') #volumes of side chains of amino acids
#aaIndexParameters.append('GRAR740102') #polarity
#aaIndexParameters.append('CHAM820101') #polarizability
#aaIndexParameters.append('ROSG850103_GUO_SASA') #solvent-accessible surface area (SASA)  Added Guo's Values to AAIndex, taken from Rose et.al 1985, from same table that provides ROSG850101 and ROSG850102, but Guo uses standard state surface area instead
#aaIndexParameters.append('GUO_NCI') #net charge index (NCI) of side chains of amino acid    Cannot find paper these came from


def AutoCovariance(fastas, aaIDs = ['GUO_H1','HOPT810101','KRIW790103','GRAR740102','CHAM820101','ROSG850103_GUO_SASA','GUO_NCI'], lag=30, deviceType='cpu'):
    return Covariance(fastas,aaIDs,lag,separate=False,calcType='AutoCovariance',deviceType=deviceType) #replace NaN with zero if sequence length <30

#if __name__ == '__main__':
    test_seq = "GIVEQCCTSICSLYQLENYCN" #insulin A (<30 length)
    test_seq = "XXDQZMNKHHH" #insulin B (>30 length)
    print (AutoCovariance(fastas = [('999', test_seq)])[1][1:])