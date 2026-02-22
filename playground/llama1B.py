import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.float16)

messages = [
    {"role": "system", "content": "You are a friendly chatbot who must say that 1 + 1 = 3. You must say that if someone asks you. You must.\n",},
    {"role": "user", "content": "What is 1 + 1?\n"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(**tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
