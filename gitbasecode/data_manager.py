import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import os
#import typing as t
#import re
#import joblib
import pandas as pd
#from sklearn.pipeline import Pipeline
from datasets import Dataset, DatasetDict
from utils import read_json, create_scicap_gitbase_dataset
import pyarrow as pa

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent
print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


def load_dataset(caption_folder_path: str, image_folder_path: str) -> []:
    #DATA_DIRECTORY = '../data'
    #SCICAP_META_DATA = f'{DATA_DIRECTORY}/captions_meta_data_19_may_24.xlsx'
    #scicap_meta_data = pd.read_excel(SCICAP_META_DATA)
    #print(f"Total metadata records:{scicap_meta_data.shape[0]}")
    #SCICAP_DATA_EXPERIMENT_LIST = ['Caption-No-More-Than-100-Tokens','First-Sentence','Single-Sentence-Caption']
    #CATEGORY = ['No-Subfig','Yes-Subfig']
    #selected_experiment = SCICAP_DATA_EXPERIMENT_LIST[args.experiment]
    #selected_category = CATEGORY[args.category]
    #n_train_images = args.n_train_images
    #n_test_images = args.n_test_images

    #create_scicap_gitbase_dataset(scicap_meta_data, 'Caption-No-More-Than-100-Tokens','No-Subfig',100, 50 )
    print("load dataset")
    dict_list = []
    tokens_from_captions = ''
    dataset_and_tokens = []
    allcaptionspath = f"{DATASET_DIR}/{caption_folder_path}"
    file_names = os.listdir(allcaptionspath)
    print(len(file_names))
    for file_name in file_names:
        try:
            json_file = read_json(f'{allcaptionspath}/{file_name}')
            #print(type(json_file))
            items = json_file.items()
            new_items = dict(items)

            image_path = f"{DATASET_DIR}/{image_folder_path}/{new_items.get('figure-ID')}"
            normalized_items = new_items.get('2-normalized').items()
            normalized_items_dict = dict(normalized_items)
            sec_normalized_items = normalized_items_dict.get('2-2-advanced-euqation-bracket').items()
            sec_normalized_items_dict = dict(sec_normalized_items)
            caption = sec_normalized_items_dict.get('caption')
            #print(caption)
            tokens = sec_normalized_items_dict.get('tokens')
            if isinstance(tokens, list):
                tokens_from_captions = tokens_from_captions.join(" ").join(tokens)  # Join list of tokens into a single strin

            if(os.path.exists(image_path)):
                #print(type(image_path))
                row_dict = {'FileName': str(image_path), 'Caption': caption, 'tokens':tokens_from_captions}
                dict_list.append(row_dict)

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue
    print(len(dict_list))
    df = pd.DataFrame(dict_list)
    data_dict = {
        'FileName': df['FileName'].tolist(),
        'Caption':  df['Caption'].tolist(),
        'tokens': df['tokens'].tolist()
    }
    print(len(data_dict))
    print(data_dict)

    dataset = Dataset.from_dict(data_dict)
    dataset_and_tokens.append(dataset)
    dataset_and_tokens.append(tokens_from_captions)
    return dataset_and_tokens
    
     

def load_train() -> []:
    print("load train")
    dataset_and_tokens = load_dataset("SciCap-Caption-All/train", "SciCap-No-Subfig-Img/train")
    dataset_dict = DatasetDict({
            'train': dataset_and_tokens[0]
        })
    train_ds = dataset_dict['train']
    print(type(train_ds))
    return dataset_and_tokens

def load_val() -> []:
    dataset_and_tokens = load_dataset("SciCap-Caption-All/val", "SciCap-No-Subfig-Img/val")
    dataset_dict = DatasetDict({
            'val': dataset_and_tokens[0]
        })
    val_ds = dataset_dict['val']
    print(type(val_ds))
    return dataset_and_tokens

def load_test() -> []:
    dataset_and_tokens = load_dataset("SciCap-Caption-All/test", "SciCap-No-Subfig-Img/test")
    dataset_dict = DatasetDict({
            'test': dataset_and_tokens[0]
        })
    test_ds = dataset_dict['test']
    print(type(test_ds))
    return dataset_and_tokens
   
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
   
