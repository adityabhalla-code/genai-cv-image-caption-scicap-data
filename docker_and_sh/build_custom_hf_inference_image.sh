#!/bin/bash


# Specify a name and a tag
algorithm_name=huggingface-pytorch-inference-extended
tag=2.1.0-transformers4.39-gpu-py38-cu113-ubuntu20.04

account=$(aws sts get-caller-identity --query Account --output text)


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${tag}"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Log into Docker
aws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -t ${algorithm_name} -f Dockerfile_inference .

docker tag ${algorithm_name} ${fullname}

docker push ${fullname}