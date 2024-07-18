from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig
from transformers import AutoProcessor , LlavaProcessor
from utils import calculate_bleu , get_original_caption , Logging
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import torch
import os
import logging

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir,exist_ok = True)
log_file_path = 'logs/scicap_logs.log'

log = Logging(log_file_path)

# FastAPI app initialization
app = FastAPI(title='SCICAP', description='scientific image captioning', version='0.0.1')
# for local 
# cache_dir = '/teamspace/studios/this_studio/.cache/huggingface/hub'
# for container 
cache_dir = os.getenv('CACHE_DIR', os.path.expanduser('~/.cache/huggingface/hub'))
hf_token = os.getenv('HF_TOKEN')

model_id = "bhalladitya/llva-1.5-7b-scicap"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
model = LlavaForConditionalGeneration.from_pretrained(model_id,quantization_config=quantization_config,
                                                      torch_dtype=torch.float16,
                                                      cache_dir = cache_dir)


tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
processor = AutoProcessor.from_pretrained(model_id,cache_dir=cache_dir)
processor.tokenizer = tokenizer





@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running"}


@app.post("/predict")
async def predict(prompt: str = Form(...), file: UploadFile = File(...)):
    try:
        log.logger.info("Received prediction request")
        image_id = file.filename
        log.logger.info(f"Received image_id:{image_id}")
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        log.logger.info(f"User prompt: {prompt}")

        print("---"*10)
        print(f"USER prompt:{prompt}")
        print("---"*10)
        prompt_text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\{prompt}\nASSISTANT:"
        print("PROMPT TEXT","---"*10)
        print(prompt_text)
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to("cuda", torch.float16)
    
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        
        prediction = processor.decode(output[0][2:], skip_special_tokens=True)
        assistant_response = prediction.split("ASSISTANT:")[-1].strip()

        original_caption = get_original_caption(image_id)
        belu_score = calculate_bleu(original_caption,assistant_response)
        output = {
                "caption": assistant_response,
                "user_prompt": prompt,
                "original_caption":original_caption,
                "belu_score":belu_score
        }
        
        log.logger.info(f"caption: {assistant_response}")
        log.logger.info(f"Prediction successful")
        return JSONResponse(content=output)
    except Exception as e:
        log.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

