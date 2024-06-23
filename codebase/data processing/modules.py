import os
import sys
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBList

from pathlib import Path

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

def PDB_to_DSSP(filename, filepath):
    p = PDBParser()
    structure = p.get_structure(str(filename), filepath)
    model = structure[0]
    dssp = DSSP(model, filepath)
    a_key = list(dssp.keys())[2]
    return dssp[a_key]


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    print(PDB_to_DSSP("1tup", "btang2/clark-scholars-ppi-predict/bin/1tup.pdb"))

    
