import torch

# import models.mistral.model as mistral_model
from models.gpt2.model_EE import GPT
from models.gpt2.model import GPTConfig

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.protocol.instruct.messages import UserMessage
from models.mistral.model import Transformer, ModelArgs

import json
import tiktoken

model_choice = "gpt2"
tokens_generated = 50
path = f"./weights/{model_choice}/EE_1_layers_middle_2"

if model_choice == "gpt2":
    GPTConfig.vocab_size = 50257
    model = GPT(GPTConfig)
    model.from_pretrained("gpt2")
    n_layer = model.config.n_layer
    
elif model_choice == "mistral":
    path = "./weights/mistral/7b-v0.3"
    with open(path+ "/params.json") as f:
        args = ModelArgs(**dict(json.load(f)))
        args.lora.enable = False
        model = Transformer(args).to(torch.bfloat16).to("cuda")
    model.from_pretrained(path + "/consolidated.safetensors")

if model_choice == "gpt2":
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}))[None, :]
    decode = lambda l: enc.decode(l[0,:].tolist())
if model_choice == "mistral":
    enc = MistralTokenizer.v3().instruct_tokenizer
    createMsg = lambda s: InstructRequest(messages = [UserMessage(role = "user", content = s)])
    encode = lambda s: torch.tensor(enc.encode_instruct(createMsg(s)).tokens, device = "cuda")
    decode = lambda l: enc.decode(l.tolist())


model.th = model.th * 0.95

if model_choice == "gpt2":
    for i in range(n_layer - 1):
        model.transformer.h[i].ee.load_state_dict(torch.load(f"{path}/layer_{i}_EE"))

inputs = "Nvidia is a great company because"

with torch.no_grad():
    output = model.generate(encode(inputs), temperature=1, max_new_tokens=tokens_generated, top_k = 10, use_EE = False)

print(decode(output))

with torch.no_grad():
    output = model.generate(encode(inputs), temperature=1, max_new_tokens=tokens_generated, top_k = 10, use_EE = True)

print(decode(output))

