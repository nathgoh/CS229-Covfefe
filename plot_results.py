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