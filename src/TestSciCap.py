from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory

data_dir = "D:\SciCap\SciCap"


test_path = data_dir + '\Test'



Test_dataset = image_dataset_from_directory(
               test_path,
                image_size=(250,250), # Resize the images to (180,180)
                batch_size=32,
                labels = 'inferred')


test_model = load_model("convnet_FigSubFig.keras")
test_loss, test_acc = test_model.evaluate(Test_dataset)
print(f"Test accuracy: {test_acc:.3f}")