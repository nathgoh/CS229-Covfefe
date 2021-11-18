import pandas as pd
import numpy as np
import data_processing
import json
import svm
import cnn
import plot_results

def main():
    # SVM classification
    # data_processing.preprocess_images(64, True, False) 
    # svm.svm_baseline(64)
    
    # CNN classification
    # data_processing.preprocess_images(128, False, True)
    history = cnn.cnn()
    plot_results.plot_accuracies(history)

if __name__ == '__main__':
    main() 