from datasets import Dataset, Features, Sequence, Value, Image , DatasetDict
from PIL import Image as PILImage

class HfDataset:
    def __init__(self,data):
        self.data = data
        # Define the schema
        self.schema = Features(
            {'messages': [{'content': [{'index': Value(dtype='int64', id=None),
                                        'text': Value(dtype='string', id=None),
                                        'type': Value(dtype='string', id=None)}],
                           'role': Value(dtype='string', id=None)}],
             'images': Sequence(feature=Image(mode=None, decode=True, id=None), length=-1, id=None)}
        )

    def return_image_as_is(self,image_path):
        return image_path

    def load_image(self,image_path):
        """Load an image from a file path into a PIL.Image.Image object."""
        try:
            return PILImage.open(image_path).convert('RGB')
        except IOError as e:
            print(f"Error loading image: {image_path} with error: {e}")
            return None



    def restructure_data(self):
        """Restructure data to load images and ensure compatibility with the dataset schema."""
        restructured = {
            'messages': [],
            'images': []
        }
        for item in self.data:
            restructured['messages'].append(item['messages'])
            loaded_images = []
            loaded_images = [self.load_image(img_path) for img_path in item['images'] if img_path]
            restructured['images'].append(loaded_images)
        return restructured

    def safe_encode(self,feature, obj):
        """Encode data safely according to the provided feature schema."""
        if obj is None:
            return None
        if isinstance(feature, Sequence) and isinstance(obj, list):
            return [self.safe_encode(feature.feature, item) for item in obj]
        if isinstance(feature, Features):
            return {k: self.safe_encode(feature[k], obj.get(k)) for k in feature}
        return obj

    def build_dataset(self):
        restructured_data = self.restructure_data()
        encoded_data = self.safe_encode(self.schema, restructured_data)
        dataset = Dataset.from_dict(encoded_data)#.cast_column("images",Image(mode=None, decode=True, id=None))
        dataset = dataset.cast_column("images", Sequence(feature=Image(mode=None, decode=True, id=None)))
        return dataset

