import pandas as pd
import numpy as np
import data_processing
import json
import svm

def main():
    # Do this if dataset and dictionary haven't been created yet
    # data_processing.preprocess_images(64)
    svm.svm_baseline()
    # print(accuracy)
    

if __name__ == '__main__':
    main() 