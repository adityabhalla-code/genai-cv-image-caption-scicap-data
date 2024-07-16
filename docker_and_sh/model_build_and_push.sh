#!/bin/bash

# Set variables
ECR_REGISTRY="644383320443.dkr.ecr.eu-north-1.amazonaws.com"
REPOSITORY_NAME="scicap-deployment"
VERSION_FILE="VERSION"

# Function to increment version
increment_version() {
    local delimiter=.
    local array=($(echo "$1" | tr $delimiter '\n'))
    local version_type=$2
    case $version_type in
        major)
            array[0]=$((array[0] + 1))
            array[1]=0
            array[2]=0
            ;;
        minor)
            array[1]=$((array[1] + 1))
            array[2]=0
            ;;
        patch)
            array[2]=$((array[2] + 1))
            ;;
    esac
    echo "${array[0]}.${array[1]}.${array[2]}"
}

# Read the current version
if [ ! -f $VERSION_FILE ]; then
    echo "0.0.0" > $VERSION_FILE
fi
CURRENT_VERSION=$(cat $VERSION_FILE)

# Get the version type (major, minor, patch) from the first script argument
VERSION_TYPE=$1
if [ -z "$VERSION_TYPE" ]; then
    echo "No version type specified. Use 'major', 'minor', or 'patch'."
    exit 1
fi

# Increment the version
NEW_VERSION=$(increment_version $CURRENT_VERSION $VERSION_TYPE)

# Save the new version
echo $NEW_VERSION > $VERSION_FILE

# Build the new Docker image
docker build -f model_deploy_docker -t ${REPOSITORY_NAME}:${NEW_VERSION} .

# Authenticate Docker to your ECR registry
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Tag the Docker image with the ECR repository URI
docker tag ${REPOSITORY_NAME}:${NEW_VERSION} ${ECR_REGISTRY}/${REPOSITORY_NAME}:${NEW_VERSION}

# Push the Docker image to ECR
docker push ${ECR_REGISTRY}/${REPOSITORY_NAME}:${NEW_VERSION}

echo "Docker image ${REPOSITORY_NAME}:${NEW_VERSION} has been built and pushed to ECR."
