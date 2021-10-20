import os
import json
from PIL import Image
import skimage.transform as transform
from skimage.io import imread
import random
import numpy as np
import pandas as pd
from collections import defaultdict

photo_dir = "./RoCoLe/Photos/"
class_path = "./RoCoLe/Annotations/RoCoLe-classes.xlsx"
json_path = "./RoCoLe/Annotations/RoCoLE-json.json"

multiclass_classification = {
    "healthy": 0,
    "rust_level_1": 1,
    "rust_level_2": 2,
    "rust_level_3": 3,
    "rust_level_4": 4,
    "red_spider_mite": 5
}

def preprocess_images(IMG_DIM = 720):

    path = os.path.abspath(os.path.join("./RoCoLe/","Resized Photos"))
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            return e
            
    multiclass_dict = defaultdict(list)
    flatten_images = []
    image_classification = []
    with open(json_path) as photo_json:
        photo_data = json.load(photo_json)
        for photo_info in photo_data:
            # Get photo information to help map multiclass_classifciation numbers
            # to their respective photos
            img_name = photo_info["External ID"]
            img_class = photo_info['Label']['classification']
            class_num = multiclass_classification[img_class]

            # Resize images so that they are all the same size (720, 720)
            image = Image.open(os.path.join(photo_dir, img_name))
            image_resize = image.resize((IMG_DIM, IMG_DIM), Image.BILINEAR)
            image_resize_name = str(class_num) + "_{}".format(img_name)
            image_resize.save(os.path.join(path, image_resize_name))
            
            # Convert into matrix and save into a dataframe
            image_array = imread(os.path.join(path, image_resize_name))
            flatten_images.append(image_array.flatten())
            image_classification.append(class_num)
            
            multiclass_dict[img_class].append(image_resize_name)
    
    df = pd.DataFrame(np.array(flatten_images))
    df['label'] = np.array(image_classification)
    df.to_pickle('image_classified_df.pkl')

    json_dict = json.dumps(multiclass_dict)
    f = open("multiclass_dict.json", "w")
    f.write(json_dict)
    f.close()

    print("Finished preprocessing Dataset.")
    
    

# def train_test_val_split(img_dict, test_size = 0.1, val_size = 0.1):
#     random.seed(1)

#     train, test, val = [], [], []

#     for c in img_dict:
#         img_names = list(set(img_dict[c]))
#         random.shuffle(img_names)

#         dataset_size = len(img_names)

#         test_split = int(test_size * len(img_names))
#         val_split = int(val_size * len(img_names))

#         test.extend(img_names[:test_split])
#         val.extend(img_names[test_split:test_split + val_split])
#         train.extend(img_names[test_split + val_split:])
        
#     return train, test, val

# width, height = image.size
# pad_width = width // 2
# pad_height = height // 2
# padding = (pad_width - IMG_DIM * 2, pad_height - IMG_DIM, pad_width + IMG_DIM * 2, pad_height + IMG_DIM * 1)
# image_resize = image.crop(padding)

