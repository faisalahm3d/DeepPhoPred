from sklearn.metrics import accuracy_score,recall_score, matthews_corrcoef, precision_score, roc_auc_score, f1_score
from sklearn.metrics import f1_score, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, roc_curve


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def performance_result(y_test,y_test_predict,y_test_prob):
  precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)
  res = "              Experimental result with combined features\n"
  res += "-----------------------------------------------------------------\n"
  res += "             independent test\n"
  res += "-----------------------------------------------------------------\n"
  res += "Accuracy:    {0:0.1f}\n".format(accuracy_score(y_test, y_test_predict)*100)
  res += "MCC:         {0:0.2f}\n".format(matthews_corrcoef(y_test, y_test_predict))
  res += "Precision:   {0:0.2f}\n".format(precision_score(y_test, y_test_predict))
  res += "Roc AUC :    {0:0.2f}\n".format(roc_auc_score(y_test, y_test_predict))
  res += "F1 score:    {0:0.2f}\n".format(f1_score(y_test, y_test_predict))
  res += "Sensitivity: {0:0.1f}\n".format(recall_score(y_test, y_test_predict)*100)
  res += "Specificity: {0:0.1f}\n".format(specificity_score(y_test, y_test_predict)*100)
  res += "PR AUC: {0:0.2f}\n".format(auc(recall, precision))

  # draw_roc_curve(y_test,y_test_predict_prob)
  print(res)
  tn, fp, fn, tp = confusion_matrix(y_test, y_test_predict).ravel()
  print("Tn: {} Fp : {} Fn : {} Tp : {}\n\n".format(tn, fp, fn , tp))