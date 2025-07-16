from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="/tmp")
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="/tmp")

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate(data: Prompt):
    input_ids = tokenizer.encode(data.prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=150,
        temperature=0.9,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": text[len(data.prompt):].strip()}
