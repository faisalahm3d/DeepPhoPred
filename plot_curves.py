from matplotlib import pyplot as plt
def plot_roc_curves(fpr, tpr, roc_auc, label,protein_site,colors):
    plt.figure()
    lw = 1.0
    #colors = ['green', 'darkorange', 'red']
    for i in range(len(label)):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='p{0} {1} (AUC = {2:0.2f})'.format(protein_site,label[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.figure(figsize=(30,10))
    plt.savefig('roc_curve_final'+protein_site+'.png', dpi=500)
    plt.show()

def plot_pr_curves(fpr, tpr, pr_auc, label, protein_site, colors):
    plt.figure()
    lw = 1.0
    #colors = ['green', 'darkorange', 'red']
    for i in range(len(label)):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw, label='p{0} {1} (AUC = {2:0.2f})'.format(protein_site,label[i], pr_auc[i]))

    #plt.plot([0, 1], [1, 0], 'k--', lw=lw)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower left")
    # plt.figure(figsize=(30,10))
    plt.savefig('pr_curve_p'+protein_site+'.png', dpi=500)
    plt.show()
