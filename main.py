import pandas as pd
import numpy as np
import data_processing
import json
import svm
import cnn

def main():
    # SVM classification
    # data_processing.preprocess_images(64, True, False) 
    # svm.svm_baseline(64)
    
    # CNN classification
    data_processing.preprocess_images(128, False, True)
    # cnn.cnn()


if __name__ == '__main__':
    main() 