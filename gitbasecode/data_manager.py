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
from utils import read_json
import pyarrow as pa

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent
print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


def load_dataset(caption_folder_path: str, image_folder_path: str) -> pd.DataFrame:
    dict_list = []
    allcaptionspath = DATASET_DIR / caption_folder_path
    file_names = os.listdir(allcaptionspath)
    print(len(file_names))
    for file_name in file_names:
        try:
            json_file = read_json(f'{allcaptionspath}/{file_name}')
            items = json_file.items()
            #print(items)
            image_path = DATASET_DIR / image_folder_path / items.mapping.get('figure-ID')
            caption = items.mapping.get('2-normalized').items().mapping.get('2-2-advanced-euqation-bracket').items().mapping.get('caption')
            
            if(os.path.exists(image_path)):
                table = pa.table({"path": [image_path]})
                row_dict = {'FileName': table, 'Caption': caption}
                dict_list.append(row_dict)

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue
    print(len(dict_list))
    df = pd.DataFrame(dict_list)
    data_dict = {
        'FileName': df['FileName'].tolist(),
        'Caption':  df['Caption'].tolist()
    }

    dataset = Dataset.from_dict(data_dict)
    dataset_dict = DatasetDict({
        'train': dataset
    })
    train_ds = dataset_dict['train']
    train_ds

def load_train() -> pd.DataFrame:
    return load_dataset("SciCap-Caption-All/train", "SciCap-No-Subfig-Img/train")

def load_val() -> pd.DataFrame:
    return load_dataset("SciCap-Caption-All/val", "SciCap-No-Subfig-Img/val")

def load_test() -> pd.DataFrame:
    return load_dataset("SciCap-Caption-All/test", "SciCap-No-Subfig-Img/test")
   
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