#!/bin/bash

# Define variables
DIRECTORY="notebooks"
LOCAL_FOLDER="extracted_files"
TARBALL_NAME="model.tar.gz"
S3_BUCKET="scicap-project"
S3_PATH="models/fine-tuned/"

# Navigate to the directory
cd $DIRECTORY

# Create the tarball
tar -czvf $TARBALL_NAME -C $LOCAL_FOLDER .

# Upload the tarball to S3
aws s3 cp $TARBALL_NAME s3://$S3_BUCKET/$S3_PATH

# Clean up by removing the local tarball
rm $TARBALL_NAME

echo "Tarball $TARBALL_NAME created and uploaded to s3://$S3_BUCKET/$S3_PATH successfully."
