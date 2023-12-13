import os 
import pickle
import json
import json
import os
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
import argparse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Show Model")
parser.add_argument("--train", type=str, help="Show Dataset File")
parser.add_argument("--val", type=str, help="Show Dataset File")
parser.add_argument("--output", type=str, help="Show Fined Tuned Folder")

args = parser.parse_args()

for arg_name in vars(args):
    arg_value = getattr(args, arg_name)
    print(f'{arg_name}: {arg_value}')


MODEL_NAME = args.model
TRAIN = args.train 
VAL = args.val
OUTPUT_DIR = args.output

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_dataset = load_dataset("json", data_files=TRAIN)["train"]
valid_dataset = load_dataset("json", data_files=VAL)["train"]

train_dataset = train_dataset.select(range(9800))
valid_dataset = valid_dataset.select(range(800))

print(len(train_dataset), len(valid_dataset))


compute_dtype = getattr(torch, "float16")


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # load_in_8bit=True,
    # quantization_config=quant_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


peft_params = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_params)
model.print_trainable_parameters()

EPOCHS = 3

BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300

training_params = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=0.2,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=10,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=0.2,
    save_steps=0.2,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    logging_dir=OUTPUT_DIR,
    report_to="wandb"
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=peft_params,
    dataset_text_field="prompt",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

