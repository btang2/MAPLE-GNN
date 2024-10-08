#OPEN-SOURCE CODE taken from existing codebase at https://github.com/ShubhrangshuGhosh2000/mat_p2ip_prj

import gzip
import sys, os
import torch

# #add parent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(currentdir)
parentDir = os.path.dirname(currentdir) + '/'
currentDir = currentdir+ '/'

from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from helpers import PPIPUtils

def readFasta(fname):
    if fname[-3:] == '.gz':
        gz = True
        f = gzip.open(fname,'rb')
    else:
        gz = False
        f = open(fname)

    curProtID = ''
    curAASeq = ''
    retLst = []
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue #blank line, skip
        if gz:
            line = line.decode('utf-8')
        if line[0] == '>':
            if curProtID != '':
                retLst.append((curProtID,curAASeq))
            line = line[1:].split('|')
            curProtID = line[0]
            curAASeq = ''
        else:
            curAASeq += line
    f.close()
    if curProtID != '':
        retLst.append((curProtID,curAASeq))
    return retLst


def getAALookup():
    AA = 'ARNDCQEGHILKMFPSTWYV'
    AADict= {}
    for item in AA:
        AADict[item] = len(AADict)
    return AADict


def loadAAData(aaIDs):
    AADict = getAALookup()
    
    #get aa index data given aaIDs
    # aaIdx = open(currentdir+'/AAidx.txt')
    aaIdx = open(os.path.join(path_root, 'helpers\AAidx.txt'))
    aaData = []
    for line in aaIdx:
        aaData.append(line.strip())
        
    myDict = {}
    for idx in range(1,len(aaData)):
        data = aaData[idx].strip().split('\t')
        myDict[data[0]] = data[1:]

    AAProperty = []
    for item in aaIDs:
        AAProperty.append([float(j) for j in myDict[item]])
    
    return AADict, aaIDs, AAProperty


def loadPairwiseAAData(aaIDs,AADict=None):
    AADict = getAALookup()
    aaIDs = aaIDs
    AAProperty = []
    for item in aaIDs:
        if item == 'Grantham':
            # f = open(currentdir+'/Grantham.txt')
            f = open(os.path.join(path_root, 'helpers\Grantham.txt'))
        elif item == 'Schneider-Wrede':
            # f = open(currentdir+'/Schneider-Wrede.txt')
            f = open(os.path.join(path_root, 'helpers\Schneider-Wrede.txt'))
        
        data = []
        colHeaders = []
        rowHeaders = []
        for line in f:
            line = line.strip().split()
            if len(line) > 0:
                if len(colHeaders) == 0:
                    colHeaders = line
                else:
                    rowHeaders.append(line[0])
                    for i in range(1,len(line)):
                        line[i] = float(line[i])
                    data.append(line[1:])
        f.close()
        AAProperty.append(data)
    return AADict, aaIDs, AAProperty


#local descriptor 10
#splits each protein sequence into 10 parts prior to computing sequence-based values
def LDEncode10(fastas,uniqueString='_+_'):
    newFastas = []
    for item in fastas:
        name = item[0]
        st = item[1]
        intervals = [0,len(st)//4,len(st)//4*2,len(st)//4*3,len(st)]
        mappings= []
        idx = 0
        for k in range(1,5):
            for i in range(0,5-k):
                newName=name+uniqueString+str(idx)
                if i == 0 and k == 4:
                    #compute middle 75%
                    newString = st[len(st)//8:len(st)//8*7]
                else:
                    newString = st[intervals[i]:intervals[i+k]]
                newFastas.append([newName,newString])
                idx += 1
    return (newFastas,10)


def STDecode(values,parts=10,uniqueString='_+_'):
    #final data list
    valLst = []
    #remap values to original proteins
    valDict = {}
    nameOrder= []
    for line in values:
        if len(valLst) == 0:
            #header
            lst = [line[0]]
            for j in range(0,parts):
                for i in range(1,len(line)):
                    lst.append(line[i]+'_'+str(j))
            valLst.append(lst) 
            continue
        line[0] = line[0].split(uniqueString)
        realName = line[0][0]
        idx = int(line[0][1])
        if realName not in valDict:
            valDict[realName] = [None]*parts
            nameOrder.append(realName)
        valDict[realName][idx] = line[1:]
        
    #error checking, and return all data
    for item in nameOrder:
        a = [item] #name
        for i in range(0,parts):
            if valDict[item][i] is None:
                print('Error, Missing Data Decode',item,i)
                exit(42)
            a.extend(valDict[item][i])
        valLst.append(a)
    return valLst


blosumMatrix = None
def loadBlosum62():
    global blosumMatrix
    global blosumMap
    if blosumMatrix is None:
        # f = open(currentdir+'/blosum62.txt')
        f = open(os.path.join(path_root, r'helpers\blosum62.txt'))
        header = None
        blosumMatrix = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split()
            if header is None:
                header = line
                continue
            blosumMatrix.append([int(item) for item in line[1:]])
        blosumMatrix =torch.tensor(blosumMatrix)
    return blosumMatrix


def genPSSM(name,sequence,folder,eVal=0.001,num_iters=3,database='uniprotSprotFull',numThreads=4):
    database = os.path.join(folder, database)
    PPIPUtils.makeDir(folder)
    try:
        f = open(database+'.psq') #check to see if one of the database files exist
        f.close()
    except:#make database
        print('creating db')
        uniprotLoc = PPIPUtils.getUniprotFastaLocations(False)[0] #get only the sprot database, not tremble
        # ######### MUST RUN IN THE SAME SHELL WHERE THE PROGRAM WILL RUN TO SET THE makeblastdb AND psiblast PATH 
        # export PATH=$PATH:/scratch/pralaycs/Shubh_Working_Remote/ncbi-blast-2.13.0+/bin
        # echo $PATH
        PPIPUtils.runLsts([['makeblastdb -in ' + uniprotLoc + ' -out '+database+' -dbtype prot']],[1])
    
    # PPIPUtils.makeDir(folder)
    f = open(folder+'tmp_'+str(name)+'.fasta','w')
    f.write('>'+name+'\n'+sequence)
    f.close()
    PPIPUtils.runLsts([['psiblast -db '+database+' -evalue '+str(eVal)+' -num_iterations '+str(num_iters)+' -out_ascii_pssm '+folder+name+'.pssm -query '+folder+'tmp_'+str(name)+'.fasta'+' -num_threads ' + str(numThreads)]],[1])
    os.remove(folder+'tmp_'+str(name)+'.fasta')


#currently doesn't handle rare letters, such as b and x, well.  May adjust later
def loadPSSM(name,sequence,folder,letters='ARNDCQEGHILKMFPSTWYV',secondEntry=False,usePSIBlast=True,eVal=0.001,num_iters=3,database='uniprotSprotFull',numThreads=4):
    lineIdx = 0
    letterMap = [-1]*len(letters)
    letterLookup = {}
    for item in letters:
        letterLookup[item] = len(letterLookup)
    try:
        f = open(folder+name+'.pssm')
    except:
        if usePSIBlast:
            genPSSM(name,sequence,folder,eVal,num_iters,database)
        try:
            f = open(folder+name+'.pssm')
        except:
            #if psiblast finds no hits, for now, use log odds from blosum matrix
            print('no hits',folder,name)#,sequence)
            blosumHeader = 'ARNDCQEGHILKMFPSTWYVBZX*'
            blosum = loadBlosum62()
            for i in range(0,len(blosumHeader)):
                if blosumHeader[i] in letterLookup:
                    letterMap[letterLookup[blosumHeader[i]]] = i
            
            letters = []
            for i in range(0,len(sequence)):
                letters.append(letterLookup.get(sequence[i],-1)) #map to final row/column if letter not in our mapping
                
            letters = torch.tensor(letters)
            pssmMat = blosum[letters,:]
            pssmMat = pssmMat[:,letterMap].tolist()
            return pssmMat

    pssmMat = []
    for line in f:
        if lineIdx <2:
            lineIdx += 1
            continue
        line = line.strip().split()
        if lineIdx == 2: #line that provides order in which letters appear
            for i in range(0,len(line)):
                #if column we are looking for, grab it
                #if secondEntry is false, the only grab column if we do not already have it
                if line[i] in letters and (letterMap[letterLookup[line[i]]] == -1 or secondEntry):
                    letterMap[letterLookup[line[i]]] = i
            lineIdx += 1
            continue
        if lineIdx < 3+len(sequence): #get sequence data
            #validation
            if int(line[0]) != lineIdx-2 or line[1] != sequence[lineIdx-3]:
                print('possible error',name,lineIdx-3,sequence[lineIdx-3],line[0],line[1])
            line = line[2:]
            data = [float(k) for k in line] + [0] #add column of zeros to end of matrix
            pssmMat.append(data)
            lineIdx += 1
            
    f.close()
    pssmMat = torch.tensor(pssmMat)
    finalLets = torch.tensor(letterMap)
    finalLets[finalLets==-1] = pssmMat.shape[0]-1 #map unknown letters to zeros at end of matrix
    retMat = pssmMat[:,finalLets].tolist()
    return retMat


#currently doesn't handle rare letters, such as b and x, well.  May adjust later
def calcBlosum62(name,sequence,letters='ARNDCQEGHILKMFPSTWYV'):
    letterMap = [-1]*len(letters)
    letterLookup = {}
    for item in letters:
        letterLookup[item] = len(letterLookup)
    
    #use log odds from blosum matrix
    blosumHeader = 'ARNDCQEGHILKMFPSTWYVBZX*'
    blosum = loadBlosum62()
    for i in range(0,len(blosumHeader)):
        if blosumHeader[i] in letterLookup:
            letterMap[letterLookup[blosumHeader[i]]] = i
    
    letters = []
    for i in range(0,len(sequence)):
        letters.append(letterLookup.get(sequence[i],-1)) #map to final row/column if letter not in our mapping
        
    letters = torch.tensor(letters)
    blosumMat = blosum[letters,:]
    blosumMat = blosumMat[:,letterMap].tolist()
    return blosumMat
