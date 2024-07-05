import matplotlib.pyplot as plt
import json
import ast
import os
import re

def get_data_version(directory_path):
    version_pattern = re.compile(r'data_v(\d+)')
    max_version = 0

    # List all items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        # Check if the item is a directory and matches the version pattern
        if os.path.isdir(item_path):
            match = version_pattern.match(item)
            if match:
                version_number = int(match.group(1))
                if version_number > max_version:
                    max_version = version_number

    # Compute the next version
    next_version = max_version + 1
    return f'data_v{next_version}'


def prepare_scicap_image_caption_list(meta_data):
    image_and_caption_list = []
    for i, row in meta_data.iterrows():
        fig_id = row['figure-ID']
        fig_path = find_fig_path(fig_id)
        caption = ast.literal_eval(row['1-lowercase-and-token-and-remove-figure-index'])['caption']
        data_point = {'image_path': fig_path, 'caption': caption}
        image_and_caption_list.append(data_point)
    return image_and_caption_list

def write_json(lst,path):
    with open(path, 'w') as json_file:
        json.dump(lst, json_file, indent=4)

def read_json(path):
    with open(path,'r') as f:
        json_f = json.load(f)
        return json_f

def plot_figure(fig_path):
    print(f"Plotting figure from path: {fig_path}")  # Debug print
    img = plt.imread(fig_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def find_fig_path(fig_id):
    current_dir = os.getcwd()
    # data_dir = os.path.join(current_dir, '..', 'scicap-data')
    # data_dir = os.path.abspath(data_dir)
    data_dir = "scicap-data"
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


def save_messages(data,output_file):
    with open(output_file, 'w') as f:
        for entry in data:
            json.dump({
                "messages": entry["messages"],
                "images": entry["images"][0].filename
            }, f)
            f.write('\n')


from PIL import Image as PILImage


def build_messages(scientific_images_and_captions):
    train_data = []
    for item in scientific_images_and_captions:
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Can you provide a scientific caption for this image?\n"},
                        {"index": 0, "type": "image", "text": None}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"index": None, "type": "text", "text": item["caption"]}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"index": None, "type": "text", "text": "What are the key features observed in this image?\n"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"index": None, "type": "text", "text": item["caption"]}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"index": None, "type": "text",
                         "text": "Can you provide a scientific explanation for this image?\n"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"index": None, "type": "text", "text": item["caption"]}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"index": None, "type": "text",
                         "text": "What is the significance of the findings shown in this image?\n"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"index": None, "type": "text", "text": item["caption"]}
                    ]
                }
            ],
            # "images": [PILImage.open(item["image_path"])]
            "images": [item["image_path"]]

        }
        train_data.append(conversation)
    return train_data
