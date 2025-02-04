python 3.11.11
pip 24.2

# Fine-Tuning LLaMA 3.2 3B with Unsloth

This repository provides a script for fine-tuning the LLaMA 3.2 3B-Instruct model using the Unsloth library. The model is fine-tuned on a dataset in JSONL format and saved in GGUF format for optimized inference.

## Requirements

Ensure you have the following dependencies installed before running the script:

```bash
pip install -r requiremets.txt
```

## Script Overview

The script performs the following steps:

1. **Load the Model and Tokenizer**
   - Loads the `unsloth/Llama-3.2-3B-Instruct` model with 4-bit quantization.

2. **Prepare the Dataset**
   - Loads the dataset from `datasetfinal.jsonl`.
   - Formats the data into an Alpaca-style prompt.

3. **Apply LoRA (Low-Rank Adaptation) Optimization**
   - Uses FastLoRA for efficient fine-tuning.
   - Applies gradient checkpointing to reduce memory usage.

4. **Train the Model**
   - Uses `SFTTrainer` from `trl` for supervised fine-tuning.
   - Training configuration includes:
     - `batch_size = 2`
     - `gradient_accumulation_steps = 4`
     - `max_steps = 10`
     - `learning_rate = 2e-4`
     - `weight_decay = 0.01`
     - `fp16` or `bf16` depending on hardware support.

5. **Save the Model**
   - Saves the fine-tuned model in GGUF format with `f16` quantization.

## Usage

Run the script with:

```bash
python unsloth_fine_tune.py
```

Ensure `datasetfinal.jsonl` is available in the working directory.

## Model Output

After training, the fine-tuned model is saved in the `model` directory in GGUF format, which can be used for inference with optimized deployment tools.

## Notes
- The script is optimized for resource efficiency using Unsloth's memory-saving techniques.
- You can increase `max_steps` for longer training.
- Modify `quantization_method` in `model.save_pretrained_gguf()` for different quantization levels.

## License
This project is open-source under the MIT License.

---

Feel free to modify the script and dataset to suit your requirements!



