from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
import gc
from transformers import TrainingArguments
from datasets import load_dataset
import os

# Constants
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs"
MODEL_SAVE_DIR = "model"

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Prompt format
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Output:
{}
"""
EOS_TOKEN = tokenizer.eos_token

# Formatting function for dataset prompts
def formatting_prompts_func(data):
    instruction = data["Instruction"]
    input_data = data["Input"]
    output_data = data["Output"]
    text = ALPACA_PROMPT.format(instruction, input_data, output_data) + EOS_TOKEN
    return {"text": text}

# Load and format dataset
DATASET_URL = "datasetfinal.jsonl"
dataset = load_dataset("json", data_files={"train": DATASET_URL}, split="train")
dataset = dataset.map(formatting_prompts_func)

# Apply LoRA patching to the model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Initialize the SFT Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=10,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

# Train the model
trainer_stats = trainer.train()
print("Training stats:", trainer_stats)

# Release GPU resources
torch.cuda.empty_cache()
gc.collect()

# Move model to CPU
#model = model.to("cpu")

# Save model safely
try:
    # Ensure no GPU processes interfere with saving
    model.save_pretrained_gguf(
        save_directory=MODEL_SAVE_DIR,
        tokenizer=tokenizer,
        quantization_method="int8",
