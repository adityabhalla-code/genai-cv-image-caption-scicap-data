from utils import find_fig_path
import json
import os

json_file_path = "/Users/IISC AIMLOPS/iisc-capstone/genai-cv-image-caption-scicap-data/scicap-data/List-of-Files-for-Each-Experiments/Single-Sentence-Caption/No-Subfig/val/file_idx.json"
destination_dir = "test_images"

import shutil
# Load the JSON file
with open(json_file_path, "r") as f:
    val_file_index = json.load(f)

if isinstance(val_file_index, list):
    selected_images = val_file_index[:500]

    for image_id in selected_images:
        source_image_path = find_fig_path(image_id)
        destination_image_path = os.path.join(destination_dir, image_id)

        # Check if the source image exists
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, destination_image_path)
            print(f"Copied {image_id} to {destination_dir}")
        else:
            print(f"Image {image_id} does not exist in the source directory")

else:
    print("The JSON file does not contain a list of filenames")
