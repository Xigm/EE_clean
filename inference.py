import torch

# import models.mistral.model as mistral_model
from models.gpt2.model_EE import GPT
from models.gpt2.model import GPTConfig

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from models.mistral.model import Transformer, ModelArgs

import json
import tiktoken

model_choice = "mistral"
tokens_generated = 100

if model_choice == "gpt2":
    GPTConfig.vocab_size = 50257
    model = GPT(GPTConfig)
    model.from_pretrained("gpt2")
    
elif model_choice == "mistral":
    path = "./weights/mistral/7b-v0.3"
    with open(path+ "/params.json") as f:
        model = Transformer(ModelArgs(**dict(json.load(f))))
    model.from_pretrained(path + "/consolidated.safetensors")

if model_choice == "gpt2":
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}))[None, :]
    decode = lambda l: enc.decode(l[0,:].tolist())
if model_choice == "mistral":
    enc = MistralTokenizer.v3().instruct_tokenizer
    encode = lambda s: torch.tensor(enc.encode_instruct(s))[None, :]
    decode = lambda l: enc.decode(l.to_list())

output = model.generate(encode("Nvidia is a very famous company which produces"), max_new_tokens=tokens_generated, top_k = 20)

print(decode(output))