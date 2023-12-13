import pickle
import json
import json
import os
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
import random
import pandas as pd
from operator import itemgetter
import torch
import warnings
warnings.filterwarnings('ignore')
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer,TrainingArguments
from trl import RewardTrainer

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

#DATASET
train_data = load_dataset("json", data_files=TRAIN)
val_data = load_dataset("json", data_files=VAL)


#MODEL LOAD
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


config = LoraConfig(
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

model = get_peft_model(model, config)
model.print_trainable_parameters()

def formatting_func(examples):
    kwargs = {"padding": "max_length",
              "truncation": True,
              "max_length": 1000,
              "return_tensors": "pt"
              }

    # Prepend the prompt and a line break to the original_response and response-1 fields.
    # prompt_plus_chosen_response = "Summarize the below text.\n\n# Text: " + examples["title"] + examples["post"] + "\n\n# Summary: " + examples["preferred_summary"]
    # prompt_plus_rejected_response = "Summarize the below text.\n\n# Text: " + examples["title"] + examples["post"] + "\n\n# Summary: " + examples["rejected_summary"]
    prompt_plus_chosen_response = examples["prompt"] + examples["chosen"]
    prompt_plus_rejected_response = examples["prompt"] + examples["rejected"]

    # Then tokenize these modified fields.
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }

format_train = train_data.map(formatting_func)
format_val = val_data.map(formatting_func)

# format_train = format_train["train"].select(range(1000))
format_val = format_val["train"].select(range(50))

len(format_train), len(format_val)

EPOCHS = 3

MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4 
LEARNING_RATE = 3e-4
### Loading the TRL reward trainer and training the trainer
training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        eval_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_steps=0.2,
        save_steps=0.2,
        warmup_steps=0.5,
        optim="paged_adamw_32bit",
        logging_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        save_total_limit=3,
        report_to="wandb"
    )

# print(type(format_train["train"]), format_train["train"][0])
trainer = RewardTrainer(model=model,
                        tokenizer=tokenizer,
                        train_dataset=format_train["train"],
                        eval_dataset=format_val,
                        args= training_args,
                        # sampler="inverted" #    inverted | importance | bayesian
                        )
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

model = torch.compile(model)
print("training start....\n")
trainer.train()
model.save_pretrained(OUTPUT_DIR)