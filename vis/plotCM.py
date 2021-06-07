import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
def compute_confusion_matrix(nb_classes,dataloaders, model):
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    with torch.no_grad():
        for i, (inputs, classes,_) in enumerate(dataloaders):
            inputs = inputs.float().cuda()
            classes = classes.long().cuda()
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 2)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds.view(-1).cpu()])
            lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    # print(conf_mat.shape[0])
    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)

    return conf_mat, class_accuracy

def plot_confusion_matrix(cm, classes, address, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        cm = np.diag(np.diag(cm))
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure(figsize=(len(classes), len(classes)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if address is not None:
        plt.savefig(address)
    plt.show()
    return cm