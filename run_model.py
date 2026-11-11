from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load base Gemma model & your Hugging Face adapter
base_model_name = "google/gemma-2-2b-it"
adapter_repo = "Preciousunoka/gemma-finetuned"

print("Loading model from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_repo)

# 2. Run test prompt
prompt = "Hello! Tell me about your fine-tuned capabilities."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)

print("\nModel Response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
