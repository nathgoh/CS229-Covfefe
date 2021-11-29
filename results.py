import matplotlib.pyplot as plt
from sklearn import metrics
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Plot training and validation accuracies over epochs
def plot_accuracies(history):
    val_accuracies = [x['val_acc'] for x in history]
    train_accuracies = [x['train_acc'] for x in history]
    plt.clf()
    plt.cla()
    plt.plot(train_accuracies, color = 'red', label = 'Training Accuracy')
    plt.plot(val_accuracies, color = 'green', label = 'Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('train_val_accuracy.png')

# Plot training and validation losses over epochs   
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.clf()
    plt.cla()
    plt.plot(train_losses, color = 'red', label = 'Training Loss')
    plt.plot(val_losses, color = 'green', label = 'Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('train_val_loss.png')

def cnn_metrics(outputs, labels):
    outputs = outputs.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    f1 = metrics.f1_score(labels, outputs, average = 'macro')
    precision = metrics.precision_score(labels, outputs, average ='macro')
    recall = metrics.recall_score(labels, outputs, average ='macro')

    return {'f1': f1, 'precision': precision, 'recall': recall}

def cnn_confusion(outputs, labels):
    outputs = outputs.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    report = metrics.classification_report(labels, outputs, labels = [0, 1, 2, 3, 4, 5])
    confusion = metrics.confusion_matrix(labels, outputs, labels = [0, 1, 2, 3, 4, 5])
    sns.heatmap(confusion, annot = True, fmt = 'd')
    plt.savefig('confusion_matrix_CNN.png')

    return report