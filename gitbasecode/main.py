from data_manager import load_dataset, load_train, load_val
from gitbase import load_model_pretrained, transforms, compute_metrics, defineTrainingArgs, dotrain, generateCaption
#import mlflow

if __name__ == '__main__':
    print("Start")
    dataset_and_tokens_train = load_train()
    dataset_and_tokens_val = load_val()
    print(type(dataset_and_tokens_train[0]))
    
    print(type(dataset_and_tokens_val))
    dotrain(dataset_and_tokens_train, dataset_and_tokens_val)
    