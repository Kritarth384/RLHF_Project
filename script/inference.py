import re
import os
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from tqdm import tqdm
import json
from tqdm import tqdm
import torch

import evaluate
from torchtext.data.metrics import bleu_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)

from transformers import AutoTokenizer
import transformers
import torch
import argparse

# arguments
parser = argparse.ArgumentParser()  
parser.add_argument("--model", type=str, help="Show Model")
parser.add_argument("--test", type=str, help="Show Dataset File")
parser.add_argument("--output", type=str, help="Show Fined Tuned Folder")

args = parser.parse_args()

for arg_name in vars(args):
    arg_value = getattr(args, arg_name)
    print(f'{arg_name}: {arg_value}')

MODEL_NAME = args.model
TEST = args.test
OUTPUT_DIR = args.output


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_NAME,
    # torch_dtype=torch.float16,
    device_map="auto",
)

def sequence_inference(prompt_sample, pipeline):
    sequences = pipeline(
        prompt_sample,
        do_sample=True,
        top_k=40,   
        temperature=0.8,
        top_p=0.80,
        num_return_sequences=1,
        repetition_penalty=1.1,
        max_length=1024,
    )

    predict = sequences[0]['generated_text']
        
    return predict


with open(TEST, "r") as f:
    test = json.load(f)
f.close()


## inference 
def prompt_generator(sample):
    # prompt=f"""Summarize the below text.\n\n# Text: {sample['title'] + sample["post"]}\n\n# Summary:"""
    prompt= sample["prompt"].split("Assistant:")[0] + "Assistant:"
    return prompt 

with open(OUTPUT_DIR , "a") as f:
    for test_sample in tqdm(test[:100]):
        prompt = prompt_generator(test_sample)
        predict = sequence_inference(prompt, pipeline)

        predict_json = {
            "target": test_sample["response"],
            "predict": predict, 
        }

        json.dump(predict_json, f)
        f.write("\n")



