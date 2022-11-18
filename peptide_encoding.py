import numpy as np
X_train = []
y_train = []

all_sites = []
all_sites_label = []

ptm_site ='S'
split = 'test'
#skip_seq = [849]

WINDOW_SIZE = 21
STRUCTURAL_WINDOW_SIZE = 21


def read_pssm_file(pssmfile):
    fo = open(pssmfile, "r+")

    str = fo.name + fo.read()
    fo.close()

    str = str.split()
    p = str[0:22]
    lastpos = str.index('Lambda')
    lastpos = lastpos - (lastpos % 62) - 4
    currentpos = str.index('Last') + 62
    p_seq = ''
    plen = 0
    pssm = {}
    while (currentpos < lastpos):
        p_no = [int(i) for i in str[currentpos]]
        p_seq = p_seq + str[currentpos + 1]
        pssm[plen] = [int(i) for i in str[currentpos + 2:currentpos + 22]]
        currentpos = currentpos + 44
        plen = plen + 1
    return p_seq, pssm, plen


def find_all_indexes(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def get_structural_feature(spd_file, protein_length, ind, flatten = True):


    with open(spd_file) as fp:
        # Read 3 lines for the top padding lines
        fp.readline()
        SSpres = fp.readlines()
        SSpre_list = []
        protein_sz = protein_length
        #STRUCTURAL_WINDOW_SIZE = 20
    for val in range(ind - STRUCTURAL_WINDOW_SIZE, ind + STRUCTURAL_WINDOW_SIZE + 1):
        now = val
        if val < 0 or val >= protein_sz:
            distance = ind - val
            now = ind + distance
        row = list(map(float, SSpres[now].strip().split()[3:]))
        SSpre_list.append(row)
    return np.asarray(SSpre_list).flatten() if flatten else SSpre_list


def get_segmented_pssm_spd(protein_length, pssm, spd, positive_index, all_index):
    temp = WINDOW_SIZE + 1
    new_psitive = [ x-1 for x in positive_index]
    for idx in all_index:
        segmented_pssm = []
        for i in range (WINDOW_SIZE, -temp, -1):
            if idx-i in pssm.keys():
                segmented_pssm.append(pssm[idx-i])
            else: segmented_pssm.append(pssm[idx+i])
        spd_feature = get_structural_feature(spd, protein_length, idx)
        feature = np.concatenate((np.array(segmented_pssm).flatten(), spd_feature), axis=0)
        X_train.append(feature)
        if idx in new_psitive:
            y_train.append(1)
        else: y_train.append(0)


def preprocess(amino_acid):
    count = 0
    seq_number = 1
    while True:
        count += 1
        # Get next line from file
        line = file.readline()
        if count % 2 == 1:
            test = line.split()
            # print(test)
            positive_index = []
            for i in range(1, len(test) - 1):
                # string_index = test[i]
                positive_index.append(int(test[i][1:]))
        else:
            # if seq_number in skip_seq:
            #     seq_number +=1
            #     continue
            pssm_file_path = 'main dataset/pssm-{}-{}/pssm{}.txt'.format(ptm_site,split,str(seq_number))
            spd_file_path = 'main dataset/spd-{}-{}/pssm{}.spd3'.format(ptm_site,split,str(seq_number))
            p_seq, pssm_dic, plen_t = read_pssm_file(pssm_file_path)
            indexes = find_all_indexes(p_seq, amino_acid)
            get_segmented_pssm_spd(len(p_seq), pssm_dic, spd_file_path, positive_index, indexes)
            print(seq_number)
            seq_number += 1
        if not line:
            break


file = open('main dataset/bphos/{}-{}.txt'.format(ptm_site,split), 'r')

preprocess(ptm_site)
import numpy as np
import pandas as pd
#np.savez('unbalanced_S_test.npz',X_train,y_train)
np.savez('window_comparision/imbalanced_{}/unbalanced_{}_{}.npz'.format(WINDOW_SIZE, ptm_site, split), X_train, y_train)

pos_count = 0
neg_count = 0
for i, val in enumerate(y_train):
    if val == 1:
        pos_count += 1
    else : neg_count += 1
print(pos_count)


