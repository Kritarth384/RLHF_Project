from dataclasses import dataclass, field
from typing import Optional
import os
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, BitsAndBytesConfig, AutoModelForCausalLM


from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed


from trl.core import LengthSampler
from trl import create_reference_model
import argparse
tqdm.pandas()

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--sft_model", type=str, help="Show Model")
parser.add_argument("--rm_model", type=str, help="Show Model")
parser.add_argument("--train", type=str, help="Show Dataset File")
parser.add_argument("--val", type=str, help="Show Dataset File")
parser.add_argument("--output", type=str, help="Show Fined Tuned Folder")

args = parser.parse_args()

for arg_name in vars(args):
    arg_value = getattr(args, arg_name)
    print(f'{arg_name}: {arg_value}')

SFT_MODEL_NAME=args.sft_model
RM_MODEL_NAME = args.rm_model
TRAIN = args.train 
VAL = args.val
OUTPUT_DIR = args.output

os.makedirs(OUTPUT_DIR, exist_ok=True)

learning_rate=1.41e-5
max_ppo_epochs=1
gradient_accumulation_steps=2
mini_batch_size=1
batch_size=4

config = PPOConfig(
    model_name=SFT_MODEL_NAME,
    learning_rate=learning_rate,
    batch_size=batch_size,
    mini_batch_size=mini_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=True,
    target_kl=0.1,
    ppo_epochs=max_ppo_epochs,
    seed=42,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
)

sent_kwargs = {
    "top_k": None, # Return all scores.
    "function_to_apply": "none", # You want the raw logits without softmax.
    "batch_size": 4
}

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_NAME, device_map="auto")
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


## Building dataset
def build_dataset(
    tokenizer,
    dataset_name,
):

    # load imdb with datasets
    ds = load_dataset("json", data_files=dataset_name, split="train")
    ds = ds.select(range(1000))
    original_columns = ds.column_names
    num_proc = 24

    def tokenize(sample):
        
        # Wrap each dialogue with the instruction.
        # prompt=f"""Summarize the below text.\n\n# Text: {sample["title"]+ sample["post"]}\n\n# Summary: """
        prompt= sample["prompt"]
        sample["input_ids"] = tokenizer(prompt, padding="max_length", max_length=1000, truncation=True,return_tensors="pt").input_ids.cuda()
        sample["input_ids"] = sample["input_ids"].reshape(1000)
        # print(sample["input_ids"])
        # This must be called "query", which is a requirement of our PPO library.
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
        

    # Tokenize each dialogue.
    ds = ds.map(tokenize, batched=False, remove_columns=original_columns)
    # ds = ds.filter(lambda x: len(x["input_ids"]) < 1, batched=False)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, TRAIN)


# collator
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

set_seed(config.seed)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    SFT_MODEL_NAME,
    torch_dtype = torch.bfloat16,
    device_map='auto',
    peft_config=lora_config,
)

optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
)

ref_model = create_reference_model(model)

## PPO trainer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)


## RM model setup:
rm_tokenizer = AutoTokenizer.from_pretrained(RM_MODEL_NAME, device_map="auto")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=RM_MODEL_NAME,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.bfloat16},
    tokenizer=rm_tokenizer,
    return_token_type_ids=False,
)

## GENERATION config
generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
    # "max_new_tokens": 32, # specify how many tokens you want to generate at most
}

output_min_length = 32
output_max_length = 1000
output_length_sampler = LengthSampler(output_min_length, output_max_length)

total_steps = 250

for epoch in range(config.ppo_epochs):
    os.makedirs(OUTPUT_DIR + f"/epoch_{epoch}", exist_ok=True)
    print(len(ppo_trainer.dataloader))
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        
        question_tensors = batch["input_ids"]
        
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]).cuda() for output in pipe_outputs]
        
        # print(question_tensors[0].get_device(), response_tensors[0].get_device(), rewards[0].get_device())
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        print(f'epoch_{epoch}')
        print(f'     objective/kl: {stats["objective/kl"]}')
        print(f'     ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'     ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print('-'.join('' for x in range(100)))

        if step%20 == 0 or step == total_steps:
            os.makedirs(OUTPUT_DIR + f"/epoch_{epoch}/step_{step}", exist_ok=True)
            model.save_pretrained(OUTPUT_DIR + f"/epoch_{epoch}/step_{step}")
            tokenizer.save_pretrained(OUTPUT_DIR + f"/epoch_{epoch}/step_{step}")
    



