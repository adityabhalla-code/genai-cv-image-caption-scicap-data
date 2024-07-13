from transformers import AutoProcessor,TrainingArguments, Trainer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict
from evaluate import load
import torch
from PIL import Image
from transformers.tokenization_utils_base import BatchEncoding
from utils import get_image_mean, get_image_stddev
from peft import LoraConfig, TaskType, get_peft_model

CHECKPOINT = "microsoft/git-base"

def load_model_pretrained() -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    return processor

def transforms(batch: Dataset):
    processor =  load_model_pretrained()
    imgProcessor = processor.image_processor
    txtTokenizer = processor.tokenizer
    txtTokenizer.padding_side = "left"
    txtTokenizer.truncation_side = "left"

    imgProcessor.do_normalize
    
    imagesarr = []
    captionsarr = []
    for file_name, caption in zip(batch["FileName"], batch["Caption"]):  # Iterate over file names and captions
        # Assuming file_name is a valid path or a file-like object
        image = Image.open(file_name)
        imagesarr.append(image)
        captionsarr.append(caption)
    images = imagesarr[0]
    captions = captionsarr[0]
    mu_rgb = get_image_mean(batch)
    std_rgb = get_image_stddev(batch)
    imgProcessor.preprocess(images=imagesarr,image_mean=mu_rgb, image_std=std_rgb)

    inputs = processor(images=imagesarr, text=captionsarr, return_tensors="pt",padding=True, truncation=True)
    inputs.update({"labels": inputs["input_ids"]})
    print("transform completed")
    #print(inputs.pop("labels"))    


def compute_metrics(eval_pred, processor: AutoProcessor, compute_result):
    wer = load("wer")
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}

def modifyModel(tokens:str):
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
    # Load the tokenizer associated with the model
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    # Tokenize the input text
    inputs = tokenizer(tokens, return_tensors="pt")

    # Get the input IDs
    input_ids = inputs["input_ids"]
    target_modules = ["q_proj", "v_proj"]  # Example target modules, adjust as needed
    peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=8,
                         lora_alpha=32, lora_dropout=0.1, target_modules=target_modules)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
def defineTrainingArgs() -> TrainingArguments:
    model_name = CHECKPOINT.split("/")[1]

    training_args = TrainingArguments(
        output_dir=f"{model_name}-scicap",
        learning_rate=5e-5,
        num_train_epochs=10,
        fp16=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=50,
        remove_unused_columns=False,
        push_to_hub=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        batch_eval_metrics=True
    )
    print(training_args)
    return training_args


def dotrain(dataset_and_tokens_train, dataset_and_tokens_val):
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
    #dataset_and_tokens_train[0].set_transform(transforms(dataset_and_tokens_train[0], load_model_pretrained()))
    train_ds = dataset_and_tokens_train[0]
    train_ds.set_transform(transforms)
    print(f"type of train ds:{type(train_ds)}")
    val_ds = dataset_and_tokens_val[0]
    val_ds.set_transform(transforms)
    print(f"type of val ds:{type(val_ds)}")
    
    tokens = dataset_and_tokens_train[1]
    modifyModel(tokens)
    args = defineTrainingArgs()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    trainer.train()    
    
    
def generateCaption(image: Image, processor: AutoProcessor, model: PreTrainedModel) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)
    return generated_caption