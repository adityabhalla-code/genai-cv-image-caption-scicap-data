import json
with open('src/training_config.json', 'r') as f:
    config = json.load(f)

print(config)

data_version = config.get('data_version')
print(f"data_version:{data_version}")
epochs = config.get('epochs')
print(f"epochs:{epochs}")
train_batch_size = config.get('train_batch_size')
print(f"train_batch_size:{train_batch_size}")
learning_rate = config.get('learning_rate')
print(f"learning_rate:{learning_rate}")

lora_matrix_rank = config.get('lora_matrix_rank')
print(f"lora_matrix_rank:{lora_matrix_rank}")

lora_alpha = config.get('lora_alpha')
print(f"lora_alpha:{lora_alpha}")

lora_target_modules = config.get('lora_target_modules')
print(f"lora_target_modules:{lora_target_modules}")

model = config.get('model')
print(f"model:{model}")

fp16 = config.get('fp16')
print(f"fp16:{fp16}")

bf16 = config.get('bf16')
print(f"bf16:{bf16}")


# parser = argparse.ArgumentParser(description="Train model with specific data version")
# parser.add_argument('--data_version', type=str, required=True, help='The specific data version to pull from DVC')
# parser.add_argument('--train_batch_size', type=int, required=True, help='Training batch size')
# parser.add_argument('--lora_matrix_rank', type=int,  required=True, help='Lora matrix rank for training')
# parser.add_argument('--model', type=str,help='Base model or tuned model', choices=['base','tuned'],default='base')
# parser.add_argument('--epochs', type=int , required=True ,help='training epochs')
# args = parser.parse_args()


