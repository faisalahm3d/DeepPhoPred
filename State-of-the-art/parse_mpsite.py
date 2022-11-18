import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
ptm_residue = 'T'
dic_pred = {}

if __name__ == '__main__':
    with open("data_mpsite/State-of-the-art-{}.html".format(ptm_residue)) as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    #print(soup.prettify())
    tag = soup.table
    tr = tag.find_all("tr")[1]
    list_td = tr.find_all("td")[0]
    cols = [ele.text.strip() for ele in list_td]
    protein_id = [ele for ele in cols if ele]
    print(len(protein_id))


    list_td = tr.find_all("td")[1]
    cols = [ele.text.strip() for ele in list_td]
    position = [int(ele) for ele in cols if ele]
    print(len(position))

    list_td = tr.find_all("td")[2]
    cols = [ele.text.strip() for ele in list_td]
    threasholds = [float(ele) for ele in cols if ele]
    print(len(threasholds))


    list_td = tr.find_all("td")[3]
    cols = [ele.text.strip() for ele in list_td]
    y_predict = [ele for ele in cols if ele]
    print(len(y_predict))
    print(y_predict)
    y_predict_num = []
    for label in y_predict:
        if label =='Non {}-phosphorylation'.format(ptm_residue):
            y_predict_num.append(0)
        else:y_predict_num.append(1)
    for i in range(len(y_predict_num)):
        dic_pred[(protein_id[i], position[i])] = {
            'pred_prob': threasholds[i],
            'pred_lab': y_predict_num[i]
        }
    print(len(dic_pred.keys()))

    with open('data_mpsite/saved_dictionary_{}.pkl'.format(ptm_residue), 'wb') as f:
        pickle.dump(dic_pred, f)






