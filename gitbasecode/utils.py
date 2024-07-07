import sys
from PIL import Image
import numpy as np
from datasets import Dataset, DatasetDict
import json

def read_json(path):
    with open(path,'r') as f:
        json_f = json.load(f)
        return json_f
    
def get_image_mean(ds: Dataset) -> float:
    reqimages = []
    reqDict = dict()
    for x in ds["FileName"]:    
        try:
            f  = Image.open(x).getdata()
            reqimages.append(f)
            #print(type(f))
            #print( sys.getsizeof(f))
        except Exception as e:
            continue
        print(len(reqimages))
    rgb_values = np.concatenate([img for img in reqimages], axis=0) / 255
    mu_rgb = np.mean(rgb_values, axis=0)  # mu_rgb.shape == (3,)
    print(mu_rgb)
    mu_rgb

    
def get_image_stddev(ds: Dataset) -> float:
    reqimages = []
    reqDict = dict()
    for x in ds["FileName"]:    
        try:
            f  = Image.open(x).getdata()
            reqimages.append(f)
            #print(type(f))
            #print( sys.getsizeof(f))
        except Exception as e:
            continue
        print(len(reqimages))
    rgb_values = np.concatenate([img for img in reqimages], axis=0) / 255
    std_rgb = np.std(rgb_values, axis=0)  # std_rgb.shape == (3,)
    print(std_rgb)
    std_rgb
        