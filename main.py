import pandas as pd
import numpy as np
import data_processing
import json
import svm
import cnn
import results

def main():
    # SVM classification
    # data_processing.preprocess_images(64, True, False) 
    # svm.svm_baseline(64)
    
    # CNN classification
    # data_processing.preprocess_images(128, False, True)
    train_val_history, test_outputs, test_results = cnn.cnn()
    metrics = results.cnn_metrics(test_outputs['predicted'], test_outputs['ground truth'])
    report = results.cnn_confusion(test_outputs['predicted'], test_outputs['ground truth'])
    results.plot_accuracies(train_val_history)
    results.plot_losses(train_val_history)
    print(metrics)
    print(report)
    

if __name__ == '__main__':
    main() 