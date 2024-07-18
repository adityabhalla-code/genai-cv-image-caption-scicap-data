import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import ast
import os
import re
#import typing as t
#import re
#import joblib
import pandas as pd
#from sklearn.pipeline import Pipeline
from datasets import Dataset, DatasetDict
from utils import read_json, get_file_list_path, get_save_file_path,write_json
import pyarrow as pa
from utils import find_fig_path
from utils import get_data_version

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent
print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent

#DATASET_DIR = PACKAGE_ROOT / "datasets"
#TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


DATA_DIRECTORY = '../data'
DATA = ['train','test','val']
CATEGORY = ['No-Subfig','Yes-Subfig']
SCICAP_DATA_EXPERIMENT_LIST = ['Caption-No-More-Than-100-Tokens','First-Sentence','Single-Sentence-Caption']

data_version = get_data_version()

def prepare_scicap_image_caption_list(meta_data):
    #image_and_caption_list = []
    dict_list = []
    for i, row in meta_data.iterrows():
        fig_id = row['figure-ID']
        fig_path = find_fig_path(fig_id)
        caption = ast.literal_eval(row['1-lowercase-and-token-and-remove-figure-index'])['caption']
        #tokens = ast.literal_eval(row['1-lowercase-and-token-and-remove-figure-index'])['token']
        #row_dict = {'FileName': fig_path, 'Caption': caption, 'tokens':tokens}
        
        row_dict = {'FileName': fig_path, 'Caption': caption}
        dict_list.append(row_dict)
        
        #data_point = {'FileName': fig_id, 'caption': caption, }
        #image_and_caption_list.append(data_point)
    #return image_and_caption_list
    return dict_list
    
def create_scicap_gitbase_dataset(scicap_meta_data,experiment,category,num_images,folder):
    file_list = read_json(get_file_list_path(folder,experiment,category))
    #test_file_list = read_json(get_file_list_path("test",experiment,category))
    print(f"Total number of {folder} images with experiment {experiment}:\n {len(file_list)}")
    #print(f"Total number of test images with experiment {experiment}:\n {len(test_file_list)}")
    data = scicap_meta_data[scicap_meta_data['figure-ID'].isin(file_list)]
    #test_data = scicap_meta_data[scicap_meta_data['figure-ID'].isin(test_file_list)]

    data = data.iloc[:num_images]
    #test_data = test_data.iloc[:n_test_images]

    data_figures = data['figure-ID'].values.tolist()
    data_file_path = get_save_file_path(f"{folder}_data_figures.json")
    write_json(data_figures,data_file_path)
    #test_data_figures = test_data['figure-ID'].values.tolist()
    #test_data_file_path = get_save_file_path("test_data_figures.json")
    #write_json(test_data_figures,test_data_file_path)

    image_and_captions = prepare_scicap_image_caption_list(data)
    #test_image_and_captions = prepare_scicap_image_caption_list(test_data)
    print(len(image_and_captions))
    df = pd.DataFrame(image_and_captions)
    data_dict = {
        'FileName': df['FileName'].tolist(),
        'Caption':  df['Caption'].tolist()
        #'tokens': df['tokens'].tolist()
    }
    print(len(data_dict))
    
    dataset = Dataset.from_dict(data_dict)
    dataset_dict = DatasetDict({
        folder: dataset
    })
    ds = dataset_dict[folder]
    return dataset
    #train_data = build_messages(train_image_and_captions)
    #test_data = build_messages(test_image_and_captions)
    #train_dataset = HfDataset(train_data).build_dataset()
    #test_dataset = HfDataset(test_data).build_dataset()
    #dataset_dict = DatasetDict({
    #    "train": train_dataset,
    #    "test": test_dataset
    #})
    
    
def load_dataset(num_images, folder) -> []:
    DATA_DIRECTORY = '../data'
    SCICAP_META_DATA = f'{DATA_DIRECTORY}/captions_meta_data_19_may_24.xlsx'
    scicap_meta_data = pd.read_excel(SCICAP_META_DATA)
    print(f"Total metadata records:{scicap_meta_data.shape[0]}")
    SCICAP_DATA_EXPERIMENT_LIST = ['Caption-No-More-Than-100-Tokens','First-Sentence','Single-Sentence-Caption']
    CATEGORY = ['No-Subfig','Yes-Subfig']
    #selected_experiment = SCICAP_DATA_EXPERIMENT_LIST[args.experiment]
    #selected_category = CATEGORY[args.category]
    #n_train_images = args.n_train_images
    #n_test_images = args.n_test_images

    return create_scicap_gitbase_dataset(scicap_meta_data, 'Caption-No-More-Than-100-Tokens','No-Subfig',num_images, folder)
     

def load_train(n_train_image):
    print("load train")
    train_ds = load_dataset(n_train_image, "train")
    print(type(train_ds))
    return train_ds

def load_val(n_val_images):
    val_ds = load_dataset(n_val_images, "val")
    print(type(val_ds))
    return val_ds

def load_test(n_test_images):
    test_ds = load_dataset(n_test_images, "test")
    print(type(test_ds))
    return test_ds
   
#def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
 #   save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
 #   save_path = TRAINED_MODEL_DIR / save_file_name

 #   remove_old_pipelines(files_to_keep=[save_file_name])
 #   joblib.dump(pipeline_to_persist, save_path)


#def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

 #   file_path = TRAINED_MODEL_DIR / file_name
 #   trained_model = joblib.load(filename=file_path)
 #   return trained_model


#def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
   # do_not_delete = files_to_keep + ["__init__.py"]
   # for model_file in TRAINED_MODEL_DIR.iterdir():
   #     if model_file.name not in do_not_delete:
   #         model_file.unlink()
   
