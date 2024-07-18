#!/bin/bash

# Directory to store the model and data
DATA_DIR="data"
S3_BUCKET="scicap-project"
S3_FILE="captions_meta_data_19_may_24.xlsx"

# Create directories if they don't exist
mkdir -p $MODEL_DIR $DATA_DIR

## Check if model already exists, if not download it
#if [ ! -d "$MODEL_DIR/your_model_name" ]; then
#    git clone https://huggingface.co/your_model_name $MODEL_DIR/your_model_name
#fi

# Check if data already exists, if not download it
if [ ! -f "$DATA_DIR/$S3_FILE" ]; then
    aws s3 cp s3://$S3_BUCKET/$S3_FILE $DATA_DIR/$S3_FILE
fi
