# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:03:59 2020

@author: mayank
"""

import requests
import time
import os
import modules
import math
import torch

#not needed for the uploade

def fetch_pdb_file(pdb_id):
    the_url = "https://files.rcsb.org/download/" + pdb_id
    #the_url = "https://files.wwpdb.org/download/" + pdb_id
    page = requests.get(the_url)
    pdb_file = page.content.decode('utf-8')
    pdb_file = pdb_file.replace('\\n', '\n')
    return(pdb_file)
    
with open('list_of_prots.txt', 'r') as f:
    data_list = f.read().strip().split('\n')
    
pdb_list = []

for data in data_list:
    pdb_list.append((data.strip().split('\t')[1].lower(), data.strip().split('\t')[2]))

#os.makedirs('pdb_files/', exist_ok=True)
#torch.cuda.empty_cache()
ctr = 0
starttime = time.process_time()
#pdb_errors = [('4uzq', 'B'), ('4v6i', 'K'),  ('4v6x', 'm'), ('3j6x', 't'), ('1scg', 'A'), ('4ug0', 'o'), ('5wlc', 'W'), ('2cva', 'A'), ('1a89', 'A'), ('1lbn', 'A'), ('6fec', 'u'), ('5wyj', 'A'), ('3jco', 'l'),   ('6fyx', 'q'), ('5gm6', 'q'), ('5nrl', 'V'), ('3j6b', 'V'), ('1orz', 'A'), ('6g2i', 'K'), ('4u3m', 'i'), ('6hhq', 'L'), ('1lfs', 'A'), ('1z8e', 'A'), ('4v7o', 'W'), ('4u3m', 'L'), ('5mrc', 'Q'),  ('1uzs', 'A'), ('5t62', 'W'), ('4v91', '1'), ('5lzs', 'h'), ('3j9m', 'a'), ('2afl', 'A'), ('2ipe', 'A'), ('6bcu', 'Z'), ('6e8g', 't'), ('5yh2', 'B'), ('1po4', 'B'), ('2ipd', 'A') , ('4v98', 's')]
pdb_errors = [('2ipe', ' '), ('2ipd', ' ')]
bugged = []
num_errors = 0
for i in range(len(pdb_errors)):
    try:
        modules.ID_to_save_graph(pdb_errors[i][0], pdb_errors[i][1])
    except Exception as e:    
        print("!! PDB PROCESSING ERROR: index " + str(i) + ", id = " + str(pdb_errors[i][1]) + " -- " + str(e))
        num_errors += 1
        bugged.append(pdb_errors[i])

    if i%10 == 0:
        cur_time = time.process_time() - starttime #secs
        print("############ PROCESSED " + str(i) + " OF " + str(len(pdb_list)) + " PDBS")
        print("############ TOTAL TIME ELAPSED: " + str(math.floor(cur_time / (60.0*60.0))) + " hr " + str(math.floor((cur_time % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(cur_time % 60.0)) + " sec")
        completion = cur_time / (i+1) * len(pdb_errors)
        print("############ TIME REMAINING: " + str(math.floor(completion / (60.0*60.0))) + " hr " + str(math.floor((completion % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(completion % 60.0)) + " sec")
        remain = completion - cur_time
        print("############ ESTIMATED TIME TO COMPLETION: " + str(math.floor(remain / (60.0*60.0))) + " hr " + str(math.floor((remain % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(remain % 60.0)) + " sec")
    #if i%100 == 0:
    #    time.sleep(60)
    #if (i >= 3):
    #    break
    #if num_errors > 30: #something is seriously wrong
    #    break
print("Num Errors: " + str(num_errors))
print("Errors: " + str(bugged))

"""
no_available_pdb = ['5wp9', '3jco', '6b1t', '4v6i', '4v6x', '6fyx', '3j6x']
pdb_error = {56: '5wp9', 80: '3jco', 553: '6b1t', 642: '4v6i', 646: '4v6x', 659: '6fyx', 670: '3j6x', 796: '4ug0', 1105: '4v81', 1152: '5iy6', 1203: '5wlc', 1277: '6ar6', 1298: '5iy6', 1313: '3jcm', 1326: '3jcm', 1327: '5gm6', 1338: '1nt9', 1363: '5gmk', 1405: '4nnj', 1416: '5iy6', 1435: '6fec', 1459: '5iy6', 1472: '4m5d', 1474: '5fj8', 1481: '5wyj', 1502: '4wxx', 1544: '4ykn', 1545: '3qt1', 1546: '5gm6', 1573: '3k7a', 1575: '5iy6', 1576: '5iy6', 1596: '5nw5', 1608: '4ui9', 1629: '2eyq', 1635: '2gif', 1678: '5gm6', 1679: '5nrl', 1687: '3jcm', 1690: '6gyk', 1691: '5oqj', 1702: '5wlc', 1703: '5fj8', 1718: '3jcm', 1730: '2b8k', 1748: '5iy6', 1786: '5y88', 1794: '3j6b', 1811: '3k07', 1814: '5x6o', 1829: '6g2i', 1833: '5oqm', 1842: '3jct', 1858: '4u3m', 1878: '6hhq', 1880: '5vh9', 1888: '3jct', 1899: '5u1s', 1908: '5wlc', 1920: '5gm6', 1930: '5gm6', 1956: '5h64', 1962: '4ug0', 1972: '6ez8', 1975: '5wlc', 1994: '4v7o', 2032: '4ug0', 2033: '4u3m', 2054: '5mrc', 2118: '1xi4', 2181: '5iy6', 2182: '5iy6', 2193: '5a9q', 2205: '5t62', 2208: '5a31', 2211: '4m5d', 2214: '5gm6', 2274: '5a9q', 2319: '4ug0', 2342: '4v91', 2358: '4ug0', 2384: '4ug0', 2403: '6gmh', 2414: '5lzs', 2483: '5luq', 2497: '3j9m', 2534: '5iy6', 2535: '3ffz', 2538: '5oqj', 2544: '5csk', 2580: '5wlc', 2595: '3jct', 2596: '5oqm', 2598: '5u1s', 2610: '3jcm', 2627: '5mps', 2637: '5x6o', 2644: '6emk', 2666: '5wlc', 2667: '6eoj', 2675: '5oqm', 2676: '3jcm', 2680: '4uhw', 2687: '5y88', 2692: '5wlc', 2719: '3ksy', 2722: '5gm6', 2736: '5oqm', 2739: '5fl8', 2769: '5wlc', 2782: '3jcm', 2784: '5oqm', 2788: '5gm6', 2809: '5z56', 2847: '5np0', 2862: '5z56', 2875: '5yz0', 2876: '6bcu', 2886: '5z58', 2934: '5nug', 2950: '6dqj', 2992: '6bcu', 2996: '5z56', 2997: '5z56', 3000: '5z56', 3047: '4a08', 3069: '5l9t', 3092: '3ei1', 3094: '5oqj', 3146: '3i7o', 3163: '3i7p', 3166: '3i8e', 3268: '6gmh', 3271: '3p8c', 3273: '6e8g', 3295: '4a0c', 3361: '4rh7', 3399: '5a9q', 3401: '3i8c', 3403: '6gmh', 3406: '6gml', 3409: '5yz0', 3420: '3prx', 3426: '3ei4', 3510: '3i89', 3695: '3a6p', 3869: '4ui9', 3890: '5yzg', 3921: '5zak', 3933: '4v98', 3958: '3p8c', 4008: '3j9m'} #pdbs that didn't process right the first time (after node 670 in no_avail)
num_errors = 0
new_pdb_error = {}
pdb_error_list = list(pdb_error.keys())
for i in range(len(pdb_error_list)):
    if(os.path.exists("codebase/data/npy/" + str(pdb_error[pdb_error_list[i]]) + "-edge_feat.npy")):
        #already processed
        print("already processed file " + str(i) + "(" + str(pdb_error[pdb_error_list[i]]) + ")")
        continue
    print("attempting to process PDB " + str(pdb_error_list[i]) + ": " + str(pdb_error[pdb_error_list[i]]))
    try:
        modules.ID_to_save_graph(pdb_error[pdb_error_list[i]])
    except Exception as e:    
        print("!! PDB PROCESSING ERROR: index " + str(i) + ", id = " + str(pdb_error[pdb_error_list[i]]) + " -- " + str(e))
        num_errors += 1
        new_pdb_error[pdb_error_list[i]] = pdb_error[pdb_error_list[i]]
    #print("CURRENT ERROR LIST: ")
    if (i % 10 == 0):
        cur_time = time.process_time() - starttime #secs
        print("TOTAL TIME ELAPSED: " + str(math.floor(cur_time / (60.0*60.0))) + " hr " + str(math.floor((cur_time % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(cur_time % 60.0)) + " sec")
        completion = cur_time / (i+1) * len(pdb_error_list)
        print("TIME REMAINING: " + str(math.floor(completion / (60.0*60.0))) + " hr " + str(math.floor((completion % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(completion % 60.0)) + " sec")
        remain = completion - cur_time
        print("ESTIMATED TIME TO COMPLETION: " + str(math.floor(remain / (60.0*60.0))) + " hr " + str(math.floor((remain % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(remain % 60.0)) + " sec")
print(str(num_errors) + " errors")
print("still PDB errors (likely out of GPU memory): " + str(new_pdb_error))

start_id = 1152 #sometimes program bugs & have to debug, so timer will reset
for i in range(start_id, len(pdb_list)):
    pdb_id = pdb_list[i]
    print("[PARSING " + str(i) + " of " + str(len(pdb_list)-1) + " PDBs: " + str(pdb_id) + "]")
    
    if (not pdb_id in no_available_pdb):
        pdbfile = fetch_pdb_file(pdb_id + ".pdb")
        filename = "codebase/data/pdb/" + pdb_id + ".pdb"
        print("Writing " + filename)
        with open(filename, "w") as fd:
            fd.write(pdbfile)
    else:
        print(str(pdb_id) + " PDB not found, manually download from RCSB")
    if(os.path.exists("codebase/data/npy/" + str(pdb_id) + "-edge_feat.npy")):
        #already processed
        print("already processed file " + str(i) + "(" + str(pdb_id) + ")")
        continue
    try:
        modules.ID_to_save_graph(pdb_id) # write files
    except Exception as e:
        print("!! PDB PROCESSING ERROR: index " + str(i) + ", id = " + str(pdb_id))
        pdb_error[i] = str(pdb_id)
        print("CURRENT ERROR LIST: ")
        print(pdb_error)
    if i%10 == 0:
        cur_time = time.process_time() - starttime #secs
        print("TOTAL TIME ELAPSED: " + str(math.floor(cur_time / (60.0*60.0))) + " hr " + str(math.floor((cur_time % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(cur_time % 60.0)) + " sec")
        completion = cur_time / (i+1 - start_id) * len(pdb_list)
        print("TIME REMAINING: " + str(math.floor(completion / (60.0*60.0))) + " hr " + str(math.floor((completion % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(completion % 60.0)) + " sec")
        remain = completion - cur_time
        print("ESTIMATED TIME TO COMPLETION: " + str(math.floor(remain / (60.0*60.0))) + " hr " + str(math.floor((remain % (60.0*60.0)) / 60.0)) + " min " + str(math.floor(remain % 60.0)) + " sec")
    if i%100 == 0:
        time.sleep(60)

print("All done!")
for key in pdb_error.keys():
    print("unable to parse PDBs: index: " + str(key) + " PDB ID: " + str(pdb_error[key]))
"""