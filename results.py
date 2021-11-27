import matplotlib.pyplot as plt
from sklearn import metrics
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Plot training and validation accuracies over epochs
def plot_accuracies(history):
    val_accuracies = [x['val_acc'] for x in history]
    train_accuracies = [x['train_acc'] for x in history]
    plt.plot(train_accuracies, color = 'red', label = 'Training Accuracy')
    plt.plot(val_accuracies, color = 'green', label = 'Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

# Plot training and validation losses over epochs   
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, color = 'red', label = 'Training Loss')
    plt.plot(val_losses, color = 'green', label = 'Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

def cnn_metrics(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    f1 = metrics.f1_score(lables, preds, average = 'macro')
    precision = metrics.precision_score(labels, preds, average ='macro')
    recall = metrics.recall_score(labels, preds, average ='macro')

    return {'f1': f1, 'precision': precision, 'recall': recall}

def cnn_confusion(outputs, labels):
    report = metrics.classification_report(labels, preds, labels = [0, 1, 2, 3, 4, 5])
    confusion = metrics.confusion_matrix(labels, preds, labels = [0, 1, 2, 3, 4, 5])
    sns.heatmap(confusion, annot = True, fmt = 'd')
    plt.savefig('confusion_matrix_CNN.png')

    return report