import numpy as np

base_folder = 'data_netphosbac'
def calculate_performance(pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(len(labels)):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # entering any of the else statement means that the evaluation metric is invalid
    acc = float(tp + tn) / len(pred_y)

    if ((tp + fp) != 0):
        sensitivity = float(tp) / (tp + fn)
    else:
        sensitivity = 0

    if ((tn + fp) != 0):
        specificity = float(tn) / (tn + fp)
    else:
        specificity = 0

    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return acc, sensitivity, specificity, MCC

# y_test = np.load('data_mpsite/y_test_s.npz', allow_pickle = True)['arr_0']
# y_pred = np.load('data_mpsite/y_predict_s.npz', allow_pickle = True)['arr_0']
#
# file = np.load('data_netphosbac/y_predict_s_dic.npz.npz', allow_pickle = True)


ptm_site = 'T'
npz_file= np.load('{}/y_true_pred_{}.npz'.format(base_folder,ptm_site), allow_pickle = True)
y_test = npz_file['arr_0']
y_pred = npz_file['arr_1']
y_pred_prob = npz_file['arr_2']

accuracy, sn, sp, mcc = calculate_performance(y_pred,y_test)
res = 'Result on independent test for {}\n'.format(ptm_site)
res +='---------------------------------------------------------------------------\n'
res += 'Sensitivity : {}\nSpecificity : {}\nAccuracy : {}\nMCC : {}\n'.format(sn,sp,accuracy,mcc)
print(res)

with open('{}/results.txt'.format(base_folder,ptm_site), 'a') as f:
    f.write(res)