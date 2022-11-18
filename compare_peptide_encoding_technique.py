from keras.layers import Input, Conv1D, Dense, Activation, Dropout, MaxPooling1D, Flatten,concatenate,GlobalMaxPool1D
from keras import Model
from tensorflow.keras.optimizers import (RMSprop, Adam, SGD)
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from keras import regularizers
from keras.regularizers import (l1, l2, l1_l2)
from keras.callbacks import (EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import (RMSprop, Adam, SGD)
from keras.layers import (Input, Dense, Dropout, Flatten, BatchNormalization,
                                     Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,GlobalAveragePooling1D,
                                     LSTM, GRU, Embedding, Bidirectional, Concatenate, Multiply)
from plot_curves import plot_roc_curves, plot_pr_curves
from sklearn.metrics import precision_recall_curve
from evalution_metrics import performance_result

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score, matthews_corrcoef, precision_score, roc_auc_score, f1_score
npzfile = np.load('balance_data/train_smote_balanced_21_Y.npz', allow_pickle=True)
x_train= npzfile['arr_0']
y_train= npzfile['arr_1']
x_test = npzfile['arr_2']
y_test = npzfile['arr_3']
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=10)


from utilities import get_pssm_spd
testX1, testX2 = get_pssm_spd(x_test)

from tensorflow.keras.models import load_model
model_spd = load_model('models/best_model_spd_Y.h5')
model_pssm = load_model('models/best_model_pssm-Y.h5')
model_combined = load_model('models/best_model_combined_Y_ind.h5')

probabilities_pssm = model_pssm.predict([testX1])
prob_pssm=[]
for j in range(len(probabilities_pssm)):
    prob = probabilities_pssm[j][1]
    prob_pssm.append(prob)
predicted_classes_pssm = probabilities_pssm >= 0.5
predicted_classes_pssm = predicted_classes_pssm.astype(int)
performance_result(y_test,predicted_classes_pssm,prob_pssm)

probabilities_spd = model_spd.predict([testX2])
prob_spd=[]
for j in range(len(probabilities_spd)):
    prob = probabilities_spd[j][1]
    prob_spd.append(prob)
predicted_classes_spd = probabilities_spd >= 0.5
predicted_classes_spd = predicted_classes_spd.astype(int)
performance_result(y_test,predicted_classes_spd,prob_spd)

probabilities_combined = model_combined.predict([testX1,testX2])
prob_combined=[]
for j in range(len(probabilities_combined)):
    prob = probabilities_combined[j][1]
    prob_combined.append(prob)
predicted_classes_combined = probabilities_combined >= 0.5
predicted_classes_combined = predicted_classes_combined.astype(int)
performance_result(y_test,predicted_classes_combined,prob_combined)

import sklearn.metrics as metrics
import numpy as np
protein_site = 'Y'

fpr = dict()
tpr = dict()
roc_auc = dict()
label = ('Structural', 'Evolutionary', 'Combined')
fpr[0], tpr[0], _ = metrics.roc_curve(y_test, probabilities_spd)
roc_auc[0] = roc_auc_score(y_test, predicted_classes_spd)
fpr[1], tpr[1], _ = metrics.roc_curve(y_test, probabilities_pssm)
roc_auc[1] = roc_auc_score(y_test, predicted_classes_pssm)
fpr[2], tpr[2], _ = metrics.roc_curve(y_test, probabilities_combined)
roc_auc[2] = roc_auc_score(y_test, predicted_classes_combined)
colors = ['green','darkorange', 'red']
plot_roc_curves(fpr, tpr, roc_auc,label,protein_site,colors)


pr = dict()
re = dict()
threshold = dict()
pr_auc = dict()
label = ('Structural', 'Evolutionary', 'Combined')
pr[0], re[0], threshold[0] = precision_recall_curve(y_test, probabilities_spd)
pr_auc[0] = metrics.auc(re[0],pr[0])
pr[1], re[1], threshold[1] = precision_recall_curve(y_test, probabilities_pssm)
pr_auc[1] = metrics.auc(re[1],pr[1])
pr[2], re[2], _ = precision_recall_curve(y_test, probabilities_combined)
pr_auc[2] = metrics.auc(re[2],pr[2])
plot_pr_curves(pr, re, pr_auc,label,protein_site, colors)
