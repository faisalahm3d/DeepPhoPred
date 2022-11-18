import numpy as np
import pickle

ptm_site = 'S'
y_test = []
y_pred = []
y_pred_prob = []
base_folder = 'data_mpsite'

with open('{}/saved_dictionary_{}.pkl'.format(base_folder,ptm_site), 'rb') as f:
    loaded_dict = pickle.load(f)


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


def get_indexes(amino_acid):
    count = 0
    seq_number = 1
    with open('dataset-bphos/p{}.txt'.format(amino_acid), 'r') as file:
        while True:
            count += 1
            # Get next line from file
            line = file.readline()
            if not line:
                break
            if count % 2 == 1:
                test = line.split()
                # print(test)
                protein_id = test[0][1:]
                positive_index = []
                for i in range(1, len(test) - 1):
                    # string_index = test[i]
                    positive_index.append(int(test[i][1:]))
                # print(positive_index)
            else:
                # if seq_number in skip_sequence_T_test:
                #     seq_number+=1
                #     continue
                #pssm_file_path = 'main dataset/pssm-{}-{}/pssm{}.txt'.format(ptm_site,split,str(seq_number))
                #spd_file_path = 'main dataset/spd-{}-{}/pssm{}.spd3'.format(ptm_site,split,str(seq_number))
                # p_seq, pssm_dic, plen_t = read_pssm_file(pssm_file_path)
                indexes = find_all_indexes(line, amino_acid)
                for ind in indexes:
                    temp = ind+1
                    if temp in positive_index:
                        y_test.append(1)
                    else:y_test.append(0)
                    prob=loaded_dict[(protein_id,temp)]['pred_prob']
                    label = loaded_dict[(protein_id,temp)]['pred_lab']
                    y_pred.append(label)
                    y_pred_prob.append(prob)

                print(seq_number)
                seq_number+=1

            # if not line:
            #     break
            # print("Line{}: {}".format(count, line.strip()))
#file = open('bphos/{}-test.txt'.format(ptm_site), 'r')
get_indexes(ptm_site)
print(len(y_test))
print('Test')
print(len(y_pred))

np.savez('{}/y_true_pred_{}.npz'.format(base_folder,ptm_site), y_test, y_pred, y_pred_prob)
