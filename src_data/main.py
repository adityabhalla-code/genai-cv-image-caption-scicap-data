from utils import (find_fig_path , read_json , build_messages , save_messages , write_json ,
                            prepare_scicap_image_caption_list , get_data_version)
from huggingface_hub import HfApi, HfFolder, login
from  llava_hf_data import HfDataset
from datasets import DatasetDict
import pandas as pd
import argparse
import os

DATA_DIRECTORY = 'data'
PROJECT_NAME = 'scicap'
DATA = ['train','test','val']
HUGGING_FACE_USERNAME = 'bhalladitya'
CATEGORY = ['No-Subfig','Yes-Subfig']
SCICAP_DATA_EXPERIMENT_LIST = ['Caption-No-More-Than-100-Tokens','First-Sentence','Single-Sentence-Caption']
DATASET_WRITE_HF_TOKEN = os.getenv('DATASET_WRITE_HF_TOKEN')


data_version = get_data_version(DATA_DIRECTORY)
print(f'The next version name is: {data_version}')
os.makedirs(f"{DATA_DIRECTORY}/{data_version}",exist_ok=True)




def push_to_huggingface(dataset, repo_name):
    HfFolder.save_token(DATASET_WRITE_HF_TOKEN)
    dataset.push_to_hub(repo_name,token=DATASET_WRITE_HF_TOKEN)



def get_save_file_path(file_name):
    return f"{DATA_DIRECTORY}/{data_version}/{file_name}"
def get_file_list_path(data,experiment,category):
    return f"scicap-data/List-of-Files-for-Each-Experiments/{experiment}/{category}/{data}/file_idx.json"


def get_user_input(options, prompt):
    print(prompt)
    for i, option in enumerate(options):
        print(f"{i}: {option}")
    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if 0 <= choice < len(options):
                return choice
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def parse_args():
    parser = argparse.ArgumentParser(description="Select experiment list, category, and number of images.")

    parser.add_argument('--experiment', type=int, choices=range(len(SCICAP_DATA_EXPERIMENT_LIST)),
                        help='Select experiment.')
    parser.add_argument('--category', type=int, choices=range(len(CATEGORY)), help='Select category.')
    parser.add_argument('--n_train_images', type=int, help='Number of training images')
    parser.add_argument('--n_test_images', type=int, help='Number of testing images')

    args = parser.parse_args()

    if args.experiment is None:
        args.experiment = get_user_input(SCICAP_DATA_EXPERIMENT_LIST, "Select experiment:")

    if args.category is None:
        args.category = get_user_input(CATEGORY, "Select category:")

    if args.n_train_images is None:
        while True:
            try:
                args.n_train_images = int(input("Enter the number of training images: "))
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    if args.n_test_images is None:
        while True:
            try:
                args.n_test_images = int(input("Enter the number of testing images: "))
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    return args

def create_scicap_llava_hf_dataset(scicap_meta_data,experiment,category,n_train_image,n_test_images):
    train_file_list = read_json(get_file_list_path("train",experiment,category))
    test_file_list = read_json(get_file_list_path("test",experiment,category))
    print(f"Total number of training images with experiment {experiment}:\n {len(train_file_list)}")
    print(f"Total number of test images with experiment {experiment}:\n {len(test_file_list)}")
    train_data = scicap_meta_data[scicap_meta_data['figure-ID'].isin(train_file_list)]
    test_data = scicap_meta_data[scicap_meta_data['figure-ID'].isin(test_file_list)]

    train_data = train_data.iloc[:n_train_image]
    test_data = test_data.iloc[:n_test_images]

    train_data_figures = train_data['figure-ID'].values.tolist()
    train_data_file_path = get_save_file_path("train_data_figures.json")
    write_json(train_data_figures,train_data_file_path)
    test_data_figures = test_data['figure-ID'].values.tolist()
    test_data_file_path = get_save_file_path("test_data_figures.json")
    write_json(test_data_figures,test_data_file_path)

    train_image_and_captions = prepare_scicap_image_caption_list(train_data)
    test_image_and_captions = prepare_scicap_image_caption_list(test_data)
    train_data = build_messages(train_image_and_captions)
    test_data = build_messages(test_image_and_captions)
    train_dataset = HfDataset(train_data).build_dataset()
    test_dataset = HfDataset(test_data).build_dataset()
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    return dataset_dict

if __name__ == '__main__':
    args = parse_args()

    selected_experiment = SCICAP_DATA_EXPERIMENT_LIST[args.experiment]
    selected_category = CATEGORY[args.category]
    n_train_images = args.n_train_images
    n_test_images = args.n_test_images

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

    dataset = create_scicap_llava_hf_dataset(scicap_meta_data,selected_experiment,selected_category,n_train_images,n_test_images)
    dataset.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset")

    hf_repo_name = f"{HUGGING_FACE_USERNAME}/{PROJECT_NAME}-{selected_experiment.lower()}-{selected_category.lower()}"
    push_to_huggingface(dataset,hf_repo_name)

