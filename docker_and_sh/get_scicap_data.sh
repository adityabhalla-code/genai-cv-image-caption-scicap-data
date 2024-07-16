#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get scicap data
wget -O scicap_data.zip "https://www.dropbox.com/scl/fi/z6vs6ztqg8ijz3pzbbqsg/scicap_data.zip?rlkey=rkjqmwo3cg9mbsv2vq6hdn2bd&e=1&dl=0"

# Unzip the data
unzip scicap_data.zip -d scicap_data

# Move the data folder
mv scicap_data/scicap_data/* scicap_data/
rm -r scicap_data/scicap_data

# Rename the data folder
mv scicap_data scicap-data

# Remove zip file
rm scicap_data.zip

echo "Setup complete."
