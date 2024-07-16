import torch

# import models.mistral.model as mistral_model
from models.gpt2.model_EE import GPT
from models.gpt2.model import GPTConfig

import tiktoken

model_choice = "gpt2"
tokens_generated = 100

if model_choice == "gpt2":
    GPTConfig.vocab_size = 50257
    model = GPT(GPTConfig)
    model.from_pretrained("gpt2")
elif model_choice == "mistral":
    path = "./weights/mistral-7b-v0.3"
    # model = mistral_model.from_pretrained(path)

if model_choice == "gpt2":
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}))[None, :]
    decode = lambda l: enc.decode(l[0,:].tolist())
if model_choice == "mistral":
    enc = tiktoken.get_encoding("mistral")
    encode = lambda s: torch.tensor(enc.encode(s))[None, :]
    decode = lambda l: enc.decode(l.to_list())

output = model.generate(encode("Nvidia is a very famous company which produces"), max_new_tokens=tokens_generated, top_k = 20)

print(decode(output))