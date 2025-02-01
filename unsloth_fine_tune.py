from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import os

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Output:
{}
"""
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(data):
    # Format a single row
    Instruction = data["Instruction"]
    Input = data["Input"]
    Output = data["Output"]
    text = alpaca_prompt.format(Instruction, Input, Output) + EOS_TOKEN
    return {"text": text}

url = "datasetfinal.jsonl"
dataset = load_dataset("json", data_files = {"train" : url}, split = "train")
# Add a new column called "text" to the dataset
dataset = dataset.map(formatting_prompts_func)


# Verify the changes
# print(dataset.column_names)


print("-------------------------------------------------------------------------------------------------")

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    # max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

print("-------------------------------------------------------------------------------------------------")


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    #data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 10,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
trainer_stats = trainer.train()

print("-------------------------------------------------------------------------------------------------")
print(trainer_stats)
print("-------------------------------------------------------------------------------------------------")

model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")


# Move the model to CPU before saving
# Save the model
#model.save_pretrained_gguf("model", tokenizer, quantization_method="int8")


#model.save_pretrained_gguf("model", tokenizer, quantization_method="int8")

#model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")


#model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
