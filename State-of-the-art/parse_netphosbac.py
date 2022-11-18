import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
ptm_residue ='T'
y_pred_prob = []
y_pred = []
dic_pred = {}
if __name__ == '__main__':
    with open("data_mpsite/NetPhosBac-p{}.html".format(ptm_residue)) as fp:
        soup = BeautifulSoup(fp, 'html.parser')
        #print(soup.prettify())
    list = soup.pre.text.strip().split('>')
    for i in range (1,len(list)):
        protein = list[i]
        lines = protein.split('\n')
        for ind in range(6, len(lines)):
            peptide_info = lines[ind].split()
            if len(peptide_info) != 8:
                break
            peptide_id = peptide_info[1][0:6]
            peptide_position = int(peptide_info[2])
            peptide_residue = peptide_info[3]
            pred_prob = float(peptide_info[5])
            pred_y = peptide_info[7]
            if peptide_residue == ptm_residue:
                y_pred_prob.append(pred_prob)
                if pred_y=='.':
                    pred_label = 0
                    #y_pred.append(0)
                else: pred_label = 1
                dic_pred[(peptide_id, peptide_position)]= {
                        'pred_prob': pred_prob,
                        'pred_lab': pred_label
                    }
    print(len(dic_pred.keys()))
    #np.savez('data_netphosbac/y_predict_s_dic.npz', dic_pred)
    with open('data_netphosbac/saved_dictionary_{}.pkl'.format(ptm_residue), 'wb') as f:
        pickle.dump(dic_pred, f)
