import data_processing
import json

def main():
    if not data_processing.read_dict():
        multiclass_dict = data_processing.classify_images()
        json_dict = json.dumps(multiclass_dict)
        f = open("multiclass_dict.json", "w")
        f.write(json_dict)
        f.close()
    
    multiclass_dict = data_processing.read_dict()
    train_set, test_set, val_set = data_processing.train_test_val_split(multiclass_dict)

    
    
if __name__ == '__main__':
    main()