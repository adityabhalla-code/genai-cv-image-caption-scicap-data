import torch
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments,LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_from_disk
from transformers.trainer_utils import get_last_checkpoint
import random
import logging
import sys
import argparse
import os
import torch



LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. \
                        The assistant gives helpful, detailed, and polite answers to the user's questions. \
                        {% for message in messages %}{% if message['role'] == 'user' %}\
                        USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}\
                        {% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""


class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"][0])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    

    args, _ = parser.parse_known_args()
    
        # make sure we have required parameters to push
    if args.push_to_hub:
        if args.hub_strategy is None:
            raise ValueError("--hub_strategy is required when pushing to Hub")
        if args.hub_token is None:
            raise ValueError("--hub_token is required when pushing to Hub")

    # sets hub id if not provided
    if args.hub_model_id is None:
        args.hub_model_id = args.model_id.replace("/", "--")
    
    logger = logging.getLogger(__name__)
    
    logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",)

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")
    
    
    # model_id = "llava-hf/llava-1.5-7b-hf"
    model_id = args.model_id
    

    quantization_config = BitsAndBytesConfig(load_in_4bit=True,)

    model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                 quantization_config=quantization_config,
                                                torch_dtype=torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer = tokenizer



    data_collator = LLavaDataCollator(processor)



    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True if get_last_checkpoint(args.output_dir) is not None else False,
        report_to="tensorboard",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size =args.eval_batch_size,
        gradient_accumulation_steps=1,
        logging_steps=5,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        fp16=args.fp16,
        bf16=False,
        
        # new args 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        
        # push to hub parameters
        # push_to_hub=args.push_to_hub,
        hub_strategy=args.hub_strategy,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )


    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules="all-linear"
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
        dataset_text_field="text",  # need a dummy field
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    trainer.train()
    # push the model to hugging face
    
    # save best model, metrics and create model card
    # trainer.create_model_card(model_name=args.hub_model_id)
    # trainer.push_to_hub()
    trainer.push_to_hub(token=args.hub_token  )
    tokenizer.push_to_hub(args.hub_model_id)
    processor.push_to_hub(args.hub_model_id)

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])
    
