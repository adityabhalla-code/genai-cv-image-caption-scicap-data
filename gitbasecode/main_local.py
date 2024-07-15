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
    url = '/content/drive/MyDrive/Colab Notebooks/SciCap-No-Subfig-Img/test/2011.07019v1-Figure3-1.png'
    image1 = Image.open(url)
    generateCaptionPretrained(image1)
    generateCaption(image1)
    