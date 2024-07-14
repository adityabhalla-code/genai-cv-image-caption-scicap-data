from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model
model_filepath = r'D:\SciCap\convnet_FigSubFig.keras'
model = load_model(model_filepath)

# Function to preprocess the image
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Define the image path and target size
image_path = r'D:\SciCap\1002.0416v1-Figure2-1.png'
target_size = (250,250)  # Change this to match the input size your model expects

# Preprocess the image
preprocessed_image = preprocess_image(image_path, target_size)

# Make a prediction
predictions = model.predict(preprocessed_image)
print(predictions[0][0])

# Assuming you have a list of class labels
class_labels = ['WithoutSubFig','WithSubFig']
if predictions[0][0] > 0.5 :
    print('WithoutSubFig')
else:
    print('WithSubFig')

# Get the index of the highest probability
"""predicted_class_index = np.argmax(predictions, axis=1)[0]
print(predicted_class_index)
# Get the class label
predicted_class_label = class_labels[predicted_class_index]

print(f"Predicted class: {predicted_class_label}")"""