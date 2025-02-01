from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    model = AutoModelForCausalLM.from_pretrained("model")  # Replace with actual save directory
    tokenizer = AutoTokenizer.from_pretrained("model")
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Failed to load the model: {e}")
