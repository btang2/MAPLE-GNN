import os
import sys
import pypdb
import fastaparser
import requests as r
from Bio import SeqIO
from io import StringIO
import time

from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
#print(sys.path)

idfile = open(os.path.join(path_root, '..\\data\\unique\\unique_array.txt'), 'r')
prot_ids = idfile.read().split(' ')
prot_id = prot_ids[1]
#for prot_id in prot_ids:
url = "https://rest.uniprot.org/uniprotkb/"
response = r.post(url + prot_id + ".fasta")
raw_fasta=StringIO(''.join(response.text))
pSeq=list(SeqIO.parse(raw_fasta,'fasta'))
print(pSeq)


idfile.close()



#with open("unique_fasta.fasta", 'w') as fasta_file:
#    writer = fastaparser.
