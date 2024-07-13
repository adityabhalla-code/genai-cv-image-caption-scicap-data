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
        
def get_file_list_path(data,experiment,category):
    return f"C:/Users/DayaSatheesh1/Downloads/scicap-data1/scicap_data/List-of-Files-for-Each-Experiments/{experiment}/{category}/{data}/file_idx.json"

#def get_save_file_path(file_name):
#    return f"{DATA_DIRECTORY}/{data_version}/{file_name}"

def write_json(lst,path):
    with open(path, 'w') as json_file:
        json.dump(lst, json_file, indent=4)

#def prepare_scicap_image_caption_list(meta_data):
#    image_and_caption_list = []
#    for i, row in meta_data.iterrows():
#        fig_id = row['figure-ID']
#        fig_path = find_fig_path(fig_id)
#        caption = ast.literal_eval(row['1-lowercase-and-token-and-remove-figure-index'])['caption']
#        data_point = {'image_path': fig_path, 'caption': caption}
#        image_and_caption_list.append(data_point)
#    return image_and_caption_list


def create_scicap_gitbase_dataset(scicap_meta_data,experiment,category,n_train_image,n_test_images):
    train_file_list = read_json(get_file_list_path("train",experiment,category))
    test_file_list = read_json(get_file_list_path("test",experiment,category))
    #print(f"Total number of training images with experiment {experiment}:\n {len(train_file_list)}")
    #print(f"Total number of test images with experiment {experiment}:\n {len(test_file_list)}")
    #train_data = scicap_meta_data[scicap_meta_data['figure-ID'].isin(train_file_list)]
    #test_data = scicap_meta_data[scicap_meta_data['figure-ID'].isin(test_file_list)]

    #train_data = train_data.iloc[:n_train_image]
    #test_data = test_data.iloc[:n_test_images]

    #train_data_figures = train_data['figure-ID'].values.tolist()
    #train_data_file_path = get_save_file_path("train_data_figures.json")
    #write_json(train_data_figures,train_data_file_path)
    #test_data_figures = test_data['figure-ID'].values.tolist()
    #test_data_file_path = get_save_file_path("test_data_figures.json")
    #write_json(test_data_figures,test_data_file_path)

    #train_image_and_captions = prepare_scicap_image_caption_list(train_data)
    #test_image_and_captions = prepare_scicap_image_caption_list(test_data)
    #train_data = build_messages(train_image_and_captions)
    #test_data = build_messages(test_image_and_captions)
    #train_dataset = HfDataset(train_data).build_dataset()
    #test_dataset = HfDataset(test_data).build_dataset()
    #dataset_dict = DatasetDict({
    #    "train": train_data,
    #    "test": test_data
    #})
    #return dataset_dict

