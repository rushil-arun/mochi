# pip install accelerate
import torch
print("Loading library...")
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Finished loading.")
checkpoint = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for fp16 use `torch_dtype=torch.float16` instead
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", dtype=torch.float16)
print("Finished loading model.")
print(model.device)
inputs = tokenizer.encode("Gravity is", return_tensors="pt").to(model.device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))