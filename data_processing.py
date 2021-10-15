import os
import json
from PIL import Image
from collections import defaultdict
import random

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

def classify_images(IMG_DIM = 720):

    path = os.path.abspath(os.path.join("./RoCoLe/","Resized Photos"))
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            return e
            
    multiclass_dict = defaultdict(set)
    with open(json_path) as photo_json:
        photo_data = json.load(photo_json)
        for photo_info in photo_data:
            # Get photo information to help map multiclass_classifciation numbers
            # to their respective photos
            img_name = photo_info["External ID"]
            img_class = photo_info['Label']['classification']
            class_num = multiclass_classification[img_class]

            # Resize images so that they are all the same size (2880, 1800)
            image = Image.open(os.path.join(photo_dir, img_name))
            width, height = image.size
            pad_width = width // 2
            pad_height = height // 2
            padding = (pad_width - IMG_DIM * 2, pad_height - IMG_DIM, pad_width + IMG_DIM * 2, pad_height + IMG_DIM * 1)
            image_resize = image.crop(padding)
            image_resize_name = str(class_num) + "_{}".format(img_name)
            image_resize.save(os.path.join(path, image_resize_name))
            multiclass_dict[img_class].add(image_resize_name)
    
    return multiclass_dict

            