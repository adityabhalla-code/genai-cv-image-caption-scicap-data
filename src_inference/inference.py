import torch
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig
from PIL import Image
import json
import base64
import io

def model_fn():
    model_id = "bhalladitya/llva-1.5-7b-scicap"
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer = tokenizer

    return model, processor, tokenizer

def predict(prompt, image_base64):
    model, processor, tokenizer = model_fn()

    image_data = base64.b64decode(image_base64)
    raw_image = Image.open(io.BytesIO(image_data))

    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return processor.decode(output[0][2:], skip_special_tokens=True)

if __name__ == "__main__":
    image_path = "../test_images/1009.0870v6-Figure5-1.png"
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\give a scientific caption for this image?\nASSISTANT:"
    
    result = predict(prompt, encoded_image)
    print(result)
