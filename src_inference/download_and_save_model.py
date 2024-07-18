from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import os 

cache_dir  = '/home/ubuntu/model_cache'
hf_token = os.getenv('HF_TOKEN')
model_id = "bhalladitya/llva-1.5-7b-scicap"

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = LlavaForConditionalGeneration.from_pretrained(model_id,quantization_config=quantization_config,
                                                          torch_dtype=torch.float16,
                                                          cache_dir = cache_dir,
                                                          use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir,use_auth_token=hf_token)
    processor = AutoProcessor.from_pretrained(model_id,cache_dir=cache_dir,use_auth_token=hf_token)
    processor.tokenizer = tokenizer
        # Save the processor with the tokenizer
    processor.save_pretrained(cache_dir)
