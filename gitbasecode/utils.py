import sys
from PIL import Image
import numpy as np
from datasets import Dataset, DatasetDict
import json

import ast
import os
import re


DATA_DIRECTORY = '../data'
DATA = ['train','test','val']
CATEGORY = ['No-Subfig','Yes-Subfig']
SCICAP_DATA_EXPERIMENT_LIST = ['Caption-No-More-Than-100-Tokens','First-Sentence','Single-Sentence-Caption']

data_version = ''

def set_data_version(version):
    data_version = version

def get_data_version():
    return data_version
    
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
    return mu_rgb

    
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
    return std_rgb
        
def get_file_list_path(data,experiment,category):
    return f"../scicap_data/List-of-Files-for-Each-Experiments/{experiment}/{category}/{data}/file_idx.json"
    
def get_save_file_path(file_name):
    return f"{DATA_DIRECTORY}/{data_version}/{file_name}"

def write_json(lst,path):
    with open(path, 'w') as json_file:
        json.dump(lst, json_file, indent=4)



def find_fig_path(fig_id):
    current_dir = os.getcwd()
    # data_dir = os.path.join(current_dir, '..', 'scicap-data')
    # data_dir = os.path.abspath(data_dir)
    data_dir = "../scicap_data"
    # print(f"Data directory: {data_dir}")
    def search_directory(directory, target):
        for root, _, files in os.walk(directory):
            for file_name in files:
                if target in file_name:
                    return os.path.join(root, file_name)
        return None

    file_path = search_directory(data_dir, fig_id)
    if file_path:
        return file_path
    else:
        raise FileNotFoundError(f"Figure with id '{fig_id}' not found in data directory or subdirectories")



