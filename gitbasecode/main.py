from data_manager import load_dataset, load_train, load_val, load_test
from gitbase import load_model_pretrained, transforms, compute_metrics, defineTrainingArgs, dotrain, generateCaption, generateCaptionPretrained
from src_data.utils import write_json
from src_data.utils import get_data_version
from utils import get_save_file_path, set_data_version
import pandas as pd
import os
from PIL import Image
#import mlflow

DATA_DIRECTORY = '../data'

data_version = get_data_version(DATA_DIRECTORY)
print(f'The next version name is: {data_version}')
os.makedirs(f"{DATA_DIRECTORY}/{data_version}",exist_ok=True)
set_data_version(data_version)

if __name__ == '__main__':
    print("Start")
    
    selected_experiment = "Caption-No-More-Than-100-Tokens"
    selected_category = 'No-Subfig'
    n_train_images = 100
    n_test_images = 50

    # print(f"Selected Dataset: {selected_data}")
    # print(f"Selected Experiment: {selected_experiment}")
    # print(f"Selected Category: {selected_category}")
    # print(f"Number of Training Images: {n_train_images}")
    # print(f"Number of Testing Images: {n_test_images}")

    meta_data = {
        "Selected Experiment": selected_experiment,
        "Selected Category": selected_category,
        "Number of Training Images": n_train_images,
        "Number of Testing Images": n_test_images
    }
    write_json(meta_data,get_save_file_path("meta_data.json"))

    SCICAP_META_DATA = f'{DATA_DIRECTORY}/captions_meta_data_19_may_24.xlsx'
    scicap_meta_data = pd.read_excel(SCICAP_META_DATA)
    print(f"Total metadata records:{scicap_meta_data.shape[0]}")

    
    train_ds = load_train(1000)
    train_ds.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset/train")
    
    val_ds = load_val(500)
    val_ds.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset/val")
    
    test_ds = load_test(50)
    test_ds.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset/test")
    
    
    dotrain(train_ds, test_ds)
    url = '../scicap_data/List-of-Files-for-Each-Experiments/Caption-No-More-Than-100-Tokens/No-Subfig/test/2011.07019v1-Figure3-1.png'
    image1 = Image.open(url)
    generateCaptionPretrained(image1)
    generateCaption(image1)
    