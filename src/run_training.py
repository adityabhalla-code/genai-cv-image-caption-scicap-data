import subprocess
import sagemaker
import argparse
import boto3
import json
import time
import os 

hf_token = os.getenv('HUGGINGFACE_TOKEN')
print("HUGGINGFACE_TOKEN:", hf_token)
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set")




def configure_aws_cli():
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')
    print("AWS_ACCESS_KEY_ID:", aws_access_key_id)
    print("AWS_SECRET_ACCESS_KEY:", aws_secret_access_key)
    print("region:", region)
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials not set in environment variables")
    # Configure AWS CLI
    subprocess.run(['aws', 'configure', 'set', 'aws_access_key_id', aws_access_key_id])
    subprocess.run(['aws', 'configure', 'set', 'aws_secret_access_key', aws_secret_access_key])
    subprocess.run(['aws', 'configure', 'set', 'region', region])


        
def pull_data_from_dvc(data_version):
    try:
        result = subprocess.run(['dvc', 'pull', f'data/{data_version}'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
        print(f"Data pulled successfully from DVC for version: {data_version}.")
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode('utf-8'))
        print(f"An error occurred while pulling data from DVC for version: {data_version}.")

def upload_directory_to_s3(local_directory, s3_bucket, s3_prefix):
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_prefix, relative_path)
            s3_client.upload_file(local_path, s3_bucket, s3_path)
            print(f"Uploaded {local_path} to s3://{s3_bucket}/{s3_path}")

if __name__=="__main__":

    with open('src/training_config.json', 'r') as f:
        config = json.load(f)

    data_version = config.get('data_version')
    epochs = config.get('epochs')
    train_batch_size = config.get('train_batch_size')
    learning_rate = config.get('learning_rate')
    lora_matrix_rank = config.get('lora_matrix_rank')
    lora_alpha = config.get('lora_alpha')
    lora_target_modules = config.get('lora_target_modules')
    model = config.get('model')
    fp16 = config.get('fp16')
    bf16 = config.get('bf16')
    gradient_checkpointing = config.get('gradient_checkpointing')
    logging_steps = config.get('logging_steps')
    gradient_accumulation_steps = config.get('gradient_accumulation_steps')
    role = config.get('sagemaker_role')
    Bucket = config.get('sagemaker_bucket')
    s3_prefix = config.get('s3_prefix')
    custom_image_uri = config.get('custom_training_image_uri')
    job_name = config.get('training_job_name')
    

    print(f"SELECTED MODEL TO TRAIN:{model}")
    
    configure_aws_cli()
    pull_data_from_dvc(data_version)
    
    
    model_id = 'llava-hf/llava-1.5-7b-hf' if model == 'base' else 'bhalladitya/llva-1.5-7b-scicap'
    print(f"BASE MODEL ID:{model_id}")
    dataset_path = f'{s3_prefix}'
    upload_directory_to_s3(f"/app/data/{data_version}/", Bucket, dataset_path)
    
    
    hyperparameters={'epochs': epochs,                                    
                     'train_batch_size': train_batch_size,                      
                     # 'eval_batch_size': 8,                         
                     'learning_rate': learning_rate,                          
                     'model_id':model_id,
                     'fp16': fp16,                                  
                     'hub_model_id': 'bhalladitya/llva-1.5-7b-scicap', # The model id of the model to push to the hub
                     'hub_token': hf_token,                            # HuggingFace token to have permission to push
                     'lora_matrix_rank':lora_matrix_rank,
                     'lora_alpha':lora_alpha,
                     'lora_target_modules':lora_target_modules,
                     'bf16' : bf16,
                     'gradient_checkpointing' : gradient_checkpointing ,
                     'logging_steps' : logging_steps ,
                     'gradient_accumulation_steps':gradient_accumulation_steps
                    }
    
    from sagemaker.huggingface import HuggingFace
    
    huggingface_estimator = HuggingFace(
        entry_point          = 'train.py',        # fine-tuning script used in training jon
        source_dir           = 'src',             # directory where fine-tuning script is stored
        instance_type        = 'ml.g5.2xlarge',   # instances type used for the training job
        instance_count       = 1,                 # the number of instances used for training
        base_job_name        = job_name,          # the name of the training job
        role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
        py_version           = 'py310',           # the python version used in the training job
        hyperparameters      = hyperparameters,   # the hyperparameter used for running the training job
        image_uri=custom_image_uri,
    )

    
    s3_dataset_path = f"s3://{Bucket}/{s3_prefix}/dataset/"
    
    data = {
        'dataset': s3_dataset_path
    }
    
    huggingface_estimator.fit(data)
    print("Waiting for SageMaker training job to start...")
    sagemaker_client = boto3.client('sagemaker')
    training_job_name = huggingface_estimator.latest_training_job.name
    waiter = sagemaker_client.get_waiter('training_job_in_progress')
    waiter.wait(TrainingJobName=training_job_name)
    print("SageMaker training job has started.")