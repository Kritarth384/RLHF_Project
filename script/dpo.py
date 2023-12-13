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
    AutoPeftModelForCausalLM
)
import argparse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments


import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer

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

from datasets import load_dataset
from random import randrange


# def format_instruction(sample):
#     return {
#             "prompt": f"""Summarize the below text.\n\n# Text: {sample['title'] + sample["post"]}\n\n# Summary:""",
#             "chosen": sample["preferred_summary"],
#             "rejected": sample["rejected_summary"],
#         }
def format_instruction(sample):
    return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }


def get_data():
    train_data = load_dataset("json", data_files=TRAIN)["train"]
    val_data = load_dataset("json", data_files=VAL)["train"]
    
    train_data = train_data.select(range(3000))
    val_data = val_data.select(range(50))
    train_data = train_data.map(format_instruction)
    val_data = val_data.map(format_instruction)
    print(train_data, val_data)
    return train_data, val_data


train_data, val_data = get_data()

model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
        # device_map={'':torch.cuda.current_device()}
        
    )
model.config.use_cache = False

model_ref = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
        # device_map={'':torch.cuda.current_device()}
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
)


EPOCHS = 1

MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-4

args = TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=0.1,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    bf16=True,
    logging_steps=1,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=0.15,
    save_steps=0.15,
    # max_grad_norm=0.3,
    output_dir=OUTPUT_DIR,
    # save_total_limit=3,
    load_best_model_at_end=True,
    logging_dir=OUTPUT_DIR,
    report_to="wandb",
    run_name="dpo_llama2",
)



dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=args,
        beta=0.1,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=512,
        max_length=512,
    )

# 6. train
dpo_trainer.train()
dpo_trainer.save_model(OUTPUT_DIR)

# 7. save
OUTPUT_DIR_F = os.path.join(OUTPUT_DIR, "final_checkpoint")
dpo_trainer.model.save_pretrained(OUTPUT_DIR_F)

