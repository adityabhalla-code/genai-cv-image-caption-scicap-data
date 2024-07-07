from data_manager import load_dataset, load_train, load_val
from gitbase import load_model_pretrained, transforms, compute_metrics, defineTrainingArgs, dotrain, generateCaption

if __name__ == '__main__':
    
    train_images = load_train()
    val_images = load_val()
    dotrain(train_images, val_images)
    