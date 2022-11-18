import numpy as np
from imblearn.over_sampling import SMOTE
def train_smote_balancing(train_npz, test_npz, site):
    sm = SMOTE(random_state=2)
    x_train, y_train = sm.fit_resample(train_npz['arr_0'], train_npz['arr_1'])
    x_test, y_test = test_npz['arr_0'], test_npz['arr_1']
    np.savez('train_smote_balanced_21_'+site+'.npz', x_train, y_train, x_test, y_test)

protein_site = 'S'
train_npzfile = np.load('../imbalanced_15/unbalanced_'+protein_site+'_train.npz', allow_pickle = True)
test_npzfile = np.load('../imbalanced_15/unbalanced_'+protein_site+'_test.npz', allow_pickle = True)
train_smote_balancing(train_npzfile,test_npzfile,protein_site)